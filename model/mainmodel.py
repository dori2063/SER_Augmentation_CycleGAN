#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 23:23:25 2019

@author: Youngdo Ahn
"""

from keras.utils import multi_gpu_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from datetime import datetime
from .submodel import real_feat_extra, real_feat_shape
from .utils import random_data

hn = 1000
D_buf = 1
G_itn = 1


class CycleGAN():
    def __init__(self, lamb_cycle=10, n_img_rows=64, n_img_cols=32, n_img_chn=1, n_gpu=1,test_1D=False):
        # Input shape
        self.test_1D = test_1D
        self.img_rows = n_img_rows
        self.img_cols = n_img_cols
        self.channels = n_img_chn
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.oshape = self.img_rows*self.img_cols
        self.data_shape = (self.oshape,)#(1582,)

        # Number of filters in the first layer of G and D
        # Loss weights
        self.lambda_cycle = lamb_cycle#10.0                    # Cycle-consistency loss
        self.lambda_id = 0 * self.lambda_cycle  

        # Calculate output shape of D (PatchGAN)
        patch_r = int(self.img_rows / 2**4)
        patch_c = int(self.img_cols / 2**4)
        self.disc_patch = (patch_r, patch_c, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64
        
        optimizer_G = Adam(2e-4,0.5)  # 0.0002
        optimizer_D = Adam(1e-4,0.5)# 0.000005

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()          
        self.g_AB_tmp = self.build_generator()
        self.g_BA_tmp = self.build_generator()  
        if self.test_1D:
            self.disc_patch = (int(patch_r*patch_c/2**4),)
            self.d_A = self.build_discriminator_1d()
            self.d_B = self.build_discriminator_1d()  
            self.g_AB = self.build_generator_1d()
            self.g_BA = self.build_generator_1d()
            self.g_AB_tmp = self.build_generator_1d()
            self.g_BA_tmp = self.build_generator_1d()
        # Input images from both domains
        emo_A = Input(shape=self.data_shape)
        emo_B = Input(shape=self.data_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(emo_A)
        fake_A = self.g_BA(emo_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        emo_A_id = self.g_BA(emo_A)
        emo_B_id = self.g_AB(emo_B)

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[emo_A, emo_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        emo_A_id, emo_B_id ])
        if n_gpu>1:
            self.d_A = multi_gpu_model(self.d_A, gpus=n_gpu)
            self.d_B = multi_gpu_model(self.d_B, gpus=n_gpu)
            self.g_AB= multi_gpu_model(self.g_AB, gpus=n_gpu)
            self.g_BA= multi_gpu_model(self.g_BA, gpus=n_gpu)
            self.g_AB_tmp= multi_gpu_model(self.g_AB, gpus=n_gpu)
            self.g_BA_tmp= multi_gpu_model(self.g_BA, gpus=n_gpu)
            self.combined = multi_gpu_model(self.combined, gpus=n_gpu)
        # For the combined model we will only train the generators
        self.d_A.compile(loss='mse', #binary_crossentropy mse
            optimizer=optimizer_D,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer_D,
            metrics=['accuracy'])
        
        self.d_A.trainable = False
        self.d_B.trainable = False
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[ 1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer_G)

    def build_generator(self):
        """U-Net Generator"""
        ff_size = 3
        def conv2d(layer_input, filters, f_size=ff_size):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=ff_size, dropout_rate=0):
            """Layers used during upsampling"""
            
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0_ = Input(shape=self.data_shape)#Input(shape=(1582,))
        d0 = Reshape(target_shape=self.img_shape)(d0_) 
        
        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img_ = Conv2D(self.channels, kernel_size=ff_size, strides=1, padding='same', activation='tanh')(u4)
        output_img = Reshape(target_shape=(self.oshape,))(output_img_) 
        out_real = Lambda(real_feat_extra,real_feat_shape)(output_img)

        return Model(d0_, out_real)

    def build_generator_1d(self):
        """U-Net Generator"""
        ff_size = 3
        def conv1d(layer_input, filters, f_size=ff_size):
            """Layers used during downsampling"""
            d = Conv1D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv1d(layer_input, skip_input, filters, f_size=ff_size, dropout_rate=0):
            """Layers used during upsampling"""
            
            u = UpSampling1D(size=2)(layer_input)
            u = Conv1D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0_ = Input(shape=self.data_shape)
        d0 = Reshape(target_shape=(self.oshape,1))(d0_) 
        
        # Downsampling
        d1 = conv1d(d0, self.gf)
        d2 = conv1d(d1, self.gf*2)
        d3 = conv1d(d2, self.gf*4)
        d4 = conv1d(d3, self.gf*8)

        # Upsampling
        u1 = deconv1d(d4, d3, self.gf*4)
        u2 = deconv1d(u1, d2, self.gf*2)
        u3 = deconv1d(u2, d1, self.gf)

        u4 = UpSampling1D(size=2)(u3)
        output_img_ = Conv1D(self.channels, kernel_size=ff_size, strides=1, padding='same', activation='tanh')(u4)
        output_img = Reshape(target_shape=(self.oshape,))(output_img_) 
        out_real = Lambda(real_feat_extra,real_feat_shape)(output_img)
        return Model(d0_, out_real)
    
    def build_discriminator(self):
        ff_size = 3
        def d_layer(layer_input, filters, f_size=ff_size, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        d0_ = Input(shape=(self.oshape,))#(1582,))
        img = Reshape(target_shape=self.img_shape)(d0_) 
        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        validity_ = Conv2D(1, kernel_size=ff_size, strides=1, padding='same')(d4)

        return Model(d0_, validity_)

    def build_discriminator_1d(self):
        ff_size = 3
        def d_layer(layer_input, filters, f_size=ff_size, normalization=True):
            """Discriminator layer"""
            d = Conv1D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        d0_ = Input(shape=(self.oshape,))
        d0 = Reshape(target_shape=(self.oshape,1))(d0_) 
        d1 = d_layer(d0, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        validity_ = Conv1D(1, kernel_size=ff_size, strides=1, padding='same')(d4)
        validity = Reshape(target_shape=(int(self.oshape/2**4),))(validity_) 
        return Model(d0_, validity)    

    def train(self, epochs=100, batch_size=1, sample_interval=500, dataA=[], 
              dataB=[], idx=0, emo=0,early_stop=False):
        st_time = datetime.now()
        # Adversarial loss ground truths
        valid = 0
        fake = 0
        early_tmp = 1000# 1000 init inf loss!!
        window = -1000
        early_pat = 0
        if self.test_1D:
            valid = np.ones((1,128))
            fake = np.zeros((1,128))
        else:
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)            
        n_batches = int(min(len(dataA), len(dataB)) / batch_size) 
        nsample = n_batches * batch_size          
        # Init Histories
        history_d = []
        history_g = []
        buffer_size = D_buf
        buffer_d_A_real = []
        buffer_d_A_fake = []
        buffer_d_B_real = []
        buffer_d_B_fake = []
        idx_name = "lamb"+str(idx)+"emo"+str(emo)
        if self.test_1D:
            idx_name = "test1D"+str(idx)+"emo"+str(emo)
        for epoch in range(epochs):
            emos_A = random_data(dataA,nsample)
            emos_B = random_data(dataB,nsample)

            for batch_i in range(nsample):
                emo_A = emos_A[batch_i]
                emo_B = emos_B[batch_i]
                emo_A = np.pad(emo_A,233,'constant',constant_values=0)#np.pad(emo_A,233,'reflect')
                emo_B = np.pad(emo_B,233,'constant',constant_values=0) 
                emo_A = emo_A.reshape(1,self.oshape)
                emo_B = emo_B.reshape(1,self.oshape)
                
                self.d_A.trainable = False
                self.d_B.trainable = False
                g_loss = 0
                for g_itn in range(G_itn):
                    g_loss = self.combined.train_on_batch([emo_A, emo_B],
                                                            [valid, valid,
                                                            emo_A, emo_B,
                                                            emo_A, emo_B])
                
                # Translate images to opposite domain
                fake_B = self.g_AB.predict(emo_A)
                fake_A = self.g_BA.predict(emo_B)

                if len(buffer_d_A_real) == buffer_size:
                    buffer_d_A_real = buffer_d_A_real[1:]
                    buffer_d_A_fake = buffer_d_A_fake[1:]
                    buffer_d_B_real = buffer_d_B_real[1:]
                    buffer_d_B_fake = buffer_d_B_fake[1:]
                buffer_d_A_real.append(emo_A)
                buffer_d_A_fake.append(fake_A) 
                buffer_d_B_real.append(emo_B)         
                buffer_d_B_fake.append(fake_B)                

                self.d_A.trainable = True
                self.d_B.trainable = True
                
                for d_itn in range(len(buffer_d_A_real)):
                    dA_loss_real = self.d_A.train_on_batch(buffer_d_A_real[d_itn], valid)  
                    dA_loss_fake = self.d_A.train_on_batch(buffer_d_A_fake[d_itn], fake)                      
                    dB_loss_real = self.d_B.train_on_batch(buffer_d_B_real[d_itn], valid)
                    dB_loss_fake = self.d_B.train_on_batch(buffer_d_B_fake[d_itn], fake)                
                
                # Train the discriminators (original images = real / translated = Fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                d_loss  = 0.5 * np.add(dA_loss, dB_loss)

                elapsed_time = datetime.now() - st_time

                # Plot the progress

                history_d.append(d_loss)
                history_g.append(g_loss)
                # GAN stopping function 
                # If at save interval => save generated image samples
                #if batch_i % 1 == 0:
                if (batch_i == sample_interval) & (epoch % 1 == 0):
                    tmp_d =np.array(history_d)
                    tmp_g =np.array(history_g)
                    xd = np.mean(tmp_d[window:,[0]])
                    xg = np.mean(tmp_g[window:,[0]])
                    xa = (tmp_g[:,[1]] + tmp_g[:,[2]])/2
                    xr = (tmp_g[:,[3]] + tmp_g[:,[4]])/2
                    xi = (tmp_g[:,[5]] + tmp_g[:,[6]])/2
                    xa = np.mean(xa[window:])
                    xr = np.mean(xr[window:])
                    xi = np.mean(xi[window:])
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %03f, G loss: %03f, adv: %03f, recon: %03f, id: %03f] time: %s " \
                                                                            % ( epoch, epochs,
                                                                                batch_i, n_batches,
                                                                                xd,xg,
                                                                                xa, xr, xi,
                                                                                elapsed_time))
                    # End epochs, history save!!
                    np.save('exp/history/%s_history_d.npy' %idx_name, history_d)
                    np.save('exp/history/%s_history_g.npy' %idx_name, history_g)
                    
                    if (early_stop!=False) & (xg > early_tmp) & (epoch > 20):
                        early_pat+=1
                        if early_pat==early_stop:
                            print("early stopped! at ", epoch)
                            self.g_AB = self.g_AB_tmp
                            self.g_BA = self.g_BA_tmp
                            return
                    elif (early_stop!=False):
                        self.g_AB_tmp = self.g_AB
                        self.g_BA_tmp = self.g_BA    
                        early_pat = 0
                        early_tmp = xg
                    
    def extract_sample(self, nsample=1, dataA=[], dataB=[], y_B=0):
        emos_A = dataA
             
        fakes = []
        y_fakes = []
        for itn in range(nsample):
            tmp=emos_A[itn]
            tmp = np.pad(tmp,233,'constant',constant_values=0)
            fake_B = self.g_AB.predict(tmp.reshape(1,self.oshape))
            fake_B = fake_B.reshape(self.oshape,)
            fake_B = fake_B[233:self.oshape-233]
            fake_B = fake_B.reshape(1,1582)
            #print("IM fake_B:",fake_B.shape)
            fakes += fake_B.tolist()
            y_fakes += [y_B]
        '''
        #emos_B = random_data(dataB, nsample//3)   
        for itn in range(nsample//3):
            tmp=emos_B[itn]
            tmp = np.pad(tmp,233,'constant',constant_values=0)
            fake_A = self.g_BA.predict(tmp.reshape(1,n_tot))
            fake_A = fake_A.reshape(n_tot,)
            fake_A = fake_A[233:n_tot-233]
            fake_A = fake_A.reshape(1,1582)
            fakes += fake_A.tolist()
            y_fakes += [2]
        '''
        return fakes, y_fakes