# -*- coding: utf-8 -*-
"""
Created on Sun May 19 22:14:06 2019

@author: Youngdo Ahn

This is refered,
https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/data_loader.py
https://keraskorea.github.io/posts/2018-10-24-%EB%94%A5%EB%9F%AC%EB%8B%9D(CycleGAN)%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%B4%20Fornite%20%EB%A5%BC%20PUBG%20%EB%A1%9C%20%EB%B0%94%EA%BE%B8%EA%B8%B0/
https://datamasters.co.kr/33  GPU
https://github.com/eesungkim/Speech_Emotion_Recognition_AAE
"""

from __future__ import print_function, division

import sys
#sys.path.insert(0,'C:/Users/USER/study/thundersvm/python/thundersvm')

import numpy as np
import os

from sklearn import svm
import collections
#from thundersvm import SVC
from keras import backend as K
from keras.utils import np_utils
import model.utils as utils
from sklearn.metrics import confusion_matrix
from model.EarlyStopping_made import *
from model import infolog
from model.utils import normalize_MeanVar, normalize_MeanVar_train, unnormalize_abs_by_train
from model.utils import normalize_MeanVar_by_train, normalize_MeanVar_by_train_adt
from model.mainmodel import CycleGAN
from model.mainmodel_1582 import CycleGAN as CycleGAN_dnn
from datetime import datetime
from model.utils import random_data

ROOT_PATH = "D:/"
CSV_DIR = ROOT_PATH+"datasets/IEMOCAP/IEMOCAP_opensmile/IEMOCAP_4class_without_e"
NPY_DIR = ROOT_PATH+"datasets/IEMOCAP/IEMOCAP_opensmile/npyfiles/"
section_list = os.listdir(CSV_DIR)
_format = '%Y-%m-%d %H:%M:%S.%f'
start_time = datetime.now().strftime(_format)[:-3]
log = infolog.log
log("THIS IS 10fold and followed ORG_190522")
#tf.random.set_random_seed(1234)

'''+++++++++++++++++++++++++++++++++++++++++ MY SWITCH +++'''
exp_title = 'GD21lmb'
n_ep = 150
task = 'early'     # 
norm = 'training_adt'       # norm = 'no', 'speakers', 'train_test', 'training', 'training_adt'

log("task:%s, norm:%s" %(task, norm))
logFileName='exp/log/10f_ep'+str(n_ep)+"_"+task+"_"+norm+"_"+exp_title+".log"
utils.makedirs("exp/log/")
infolog.init(logFileName)
    
sample_per_emo = 100
n_img_rows = 64
n_img_cols = 32
n_chn = 1
n_early = 5
n_gpu = 1
lmd = 10
val_srt  = 4
val_iter = 6

def extract_code_vector(idx):
    # Call the openSMILE data
    train_mean = 0
    train_var  = 0
    train_abs  = 0
    fold = section_list[idx]
    x_train = []
    y_train = []
    for extra_fold in section_list:
        if extra_fold != (fold):
            tmp_x = np.load("%s_ops.npy"%(NPY_DIR+extra_fold))
            if norm == 'speakers':
                tmp_x = normalize_MeanVar(tmp_x)
            tmp_x = tmp_x.tolist()
            x_train += tmp_x
            
            tmp_y = np.load("%s_lab.npy"%(NPY_DIR+extra_fold))
            tmp_y = tmp_y.tolist()            
            y_train += tmp_y
    x_train = np.array(x_train)
    if norm == 'train_test' or 'training' or 'training_adt':
        x_train, train_mean, train_var, train_abs = normalize_MeanVar_train(x_train)
    x_test = np.load("%s_ops.npy"%(NPY_DIR+fold))
    if norm == 'speakers':
        x_test = normalize_MeanVar(x_test) 
    elif norm == 'train_test':
        x_test = normalize_MeanVar(x_test)    
    elif norm == 'training':
        x_test = normalize_MeanVar_by_train(x_test, train_mean, train_var, train_abs) 
    elif norm == 'training_adt':
        x_test = normalize_MeanVar_by_train_adt(x_test, train_mean, train_var) 
    y_test = np.load("%s_lab.npy"%(NPY_DIR+fold))
    # print("test,", sum(y_test==0), sum(y_test==1), sum(y_test==2), sum(y_test==3))
    y_test = y_test.tolist()

    # Separate by Emotions of Training set
    tr_ang = []
    tr_hap = []
    tr_neu = []
    tr_sad = []
    for itn in range(len(y_train)):
        tmp = x_train[itn]
        tmp = [tmp.tolist()]
        if y_train[itn] == 0:
            tr_ang += tmp
        elif y_train[itn] == 1:
            tr_hap += tmp
        elif y_train[itn] == 2:
            tr_neu += tmp
        elif y_train[itn] == 3:
            tr_sad += tmp
    tr_ang = np.array(tr_ang)
    tr_hap = np.array(tr_hap)
    tr_neu = np.array(tr_neu)
    tr_sad = np.array(tr_sad)
    add_ang = int((len(tr_neu)+len(tr_sad)+len(tr_hap)-2*len(tr_ang))//3)
    add_hap = int((len(tr_neu)+len(tr_sad)+len(tr_ang)-2*len(tr_hap))//3)
    add_sad = int((len(tr_neu)+len(tr_hap)+len(tr_ang)-2*len(tr_sad))//3)
    add_list = [add_ang, add_hap, add_sad]
    '''
    CHECK total classs numbers
    print("ADD,", add_ang, add_hap, add_sad)
    y_train = np.array(y_train)
    print("train,", sum(y_train==0), sum(y_train==1), sum(y_train==2), sum(y_train==3))
    '''

    # Train the CycleGANs per each Emotions
    dataA = tr_neu
    dataB = 0
    emos_A = random_data(dataA)
    fake_emos = []
    y_fakes   = []
    iter_emos = [0,1,3]
    n_tmp = 0
    for itn, emo in enumerate(iter_emos):
        if emo == 0:
            dataB = tr_ang
        elif emo == 1:
            dataB = tr_hap
        elif emo == 3:
            dataB = tr_sad
        gan = CycleGAN(lamb_cycle=lmd, n_img_rows=64, n_img_cols=32, n_img_chn=n_chn, n_gpu=n_gpu, test_1D=True)
        # Train Generator (CycleGAN) by emotion
        gan.train(epochs=n_ep, batch_size=1, dataA=dataA, dataB=dataB, idx=idx, emo=emo,early_stop=n_early)   
        tmpA = emos_A[n_tmp:n_tmp+add_list[itn]]   # Use for extract
        n_tmp += add_list[itn]
        # Separate B from trainset for extract 
        tmp_fake_emo, tmp_y_fakes = gan.extract_sample(nsample=add_list[itn],dataA=tmpA, dataB=dataB, y_B=emo)
        fake_emos += tmp_fake_emo
        y_fakes += tmp_y_fakes
    x_fakes = np.array(fake_emos)    
    #print("x_fake:",x_fakes)    
    if norm == 'training_adt':
        x_fakes = unnormalize_abs_by_train(x_fakes, train_abs)
        x_train = unnormalize_abs_by_train(x_train, train_abs)
    return x_train, y_train, x_test, y_test, x_fakes, y_fakes
    
def evaluate(idx, x_train, y_train, x_test, y_test, x_fakes, y_fakes):
    acc_stat = np.array([[0.,0.],[0.,0.]])
    X_train = []
    x_test = x_test
    y_test = y_test
    # Calculate score of Synthesis only & Appended.
    for itn in range(2): ### 2
        if itn==0:
            X_train         = x_fakes
            Y_train         = y_fakes
        else:
            X_train =  X_train.tolist() + x_train.tolist()
            X_train =  np.array(X_train)
            Y_train =  y_fakes + y_train   

        log("%d th, testset:%d, trainset:%d" %(idx+1, len(x_test), len(X_train)))
        
        clf = svm.SVC(kernel='rbf',gamma=0.001, C=100,cache_size=20000)
        clf.fit(X_train, Y_train)
        y_pred=clf.predict(x_test)
    
        test_weighted_accuracy=clf.score(x_test, y_test)
    
        uar=0
        cnf_matrix = confusion_matrix(y_test, y_pred)
        diag=np.diagonal(cnf_matrix)
        for index,i in enumerate(diag):
            uar+=i/collections.Counter(y_test)[index]
        test_unweighted_accuracy=uar/len(cnf_matrix)
        accuracy=[]
        accuracy.append(float(test_weighted_accuracy*100))
        accuracy.append(float(test_unweighted_accuracy*100))
        
        # Compute confusion matrix
        cnf_matrix = np.transpose(cnf_matrix)
        cnf_matrix = cnf_matrix*100 / cnf_matrix.astype(np.int).sum(axis=0)
        cnf_matrix = np.transpose(cnf_matrix).astype(float)
        cnf_matrix = np.around(cnf_matrix, decimals=1)
    
        #accuracy per class 
        conf_mat = (cnf_matrix.diagonal()*100)/cnf_matrix.sum(axis=1)
        conf_mat = np.around(conf_mat, decimals=2)
        log('[%d:augmode]===================[0%d]'%(itn, idx+1))
        log('Feature Dimension: %d'%X_train.shape[1])
        log('Confusion Matrix:\n%s'%cnf_matrix)
        log('Accuracy per classes:\n%s'%conf_mat)
        log("WAR\t\t\t:\t%.2f %%" %(test_weighted_accuracy*100))
        log("UAR\t\t\t:\t%.2f %%" %(test_unweighted_accuracy*100))
        acc_stat[itn] = np.around(np.array(accuracy),decimals=4)
    return acc_stat
    
test_one_itn = [5]*3
if __name__ == '__main__':
    acc_stat = np.array([[0.,0.],[0.,0.]])
    ii = 0
    for idx in (test_one_itn):#range(val_srt,val_iter):
        ii += 1
        x_train, y_train, x_test, y_test, x_fakes, y_fakes = extract_code_vector(idx)
        acc_stat += evaluate(idx,  x_train, y_train, x_test, y_test, x_fakes, y_fakes)
        log('[ %s ]'%(acc_stat/float(ii)))
    
log("Start TIME:%s" % start_time)
log("End   TIME:%s" % datetime.now().strftime(_format)[:-3])
    
