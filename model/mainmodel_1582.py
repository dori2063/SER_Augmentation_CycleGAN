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
        self.data_shape = (1582,)#(1582,)

        # Number of filters in the first layer of G and D
        # Loss weights
        self.lambda_cycle = lamb_cycle#10.0                    # Cycle-consistency loss
        self.lambda_id = 0.5 * self.lambda_cycle  
      
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

        def dnn_module(layer_input, h_unit):
            d = Dense(h_unit)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        d0 = Input(shape=self.data_shape)
        d1 = dnn_module(d0, hn)
        d2 = dnn_module(d1, int(hn//2))
        d3 = dnn_module(d2, int(hn//4))
        d4 = dnn_module(d3, int(hn//2))
        d5 = dnn_module(d4, hn)

        return Model(d0, d5)
    
    def build_discriminator(self):
        def dnn_module(layer_input, h_unit):
            d = Dense(h_unit)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        d0 = Input(shape=self.data_shape)
        d1 = dnn_module(d0, hn)
        d2 = dnn_module(d1, int(hn//2))
        d3 = dnn_module(d2, int(hn//4))
        d4 = dnn_module(d3, int(hn//8))
        d5 = Dense(1,activation='sigmoid')(d4)
        
        return Model(d0, d5)

    def train(self, epochs=100, batch_size=1, sample_interval=500, dataA=[], 
              dataB=[], idx=0, emo=0,early_stop=False):
        st_time = datetime.now()
        # Adversarial loss ground truths
        valid = 0
        fake = 0
        early_tmp = 1000# 1000 init inf loss!!
        window = -1000
        early_pat = 0

        valid = np.ones((batch_size,))
        fake = np.zeros((batch_size,))            
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
                emo_A = emo_A.reshape(1,self.data_shape)
                emo_B = emo_B.reshape(1,self.data_shape)
                
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
            fake_B = self.g_AB.predict(emos_A[itn])
            fake_B = fake_B.reshape(1,self.data_shape)
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