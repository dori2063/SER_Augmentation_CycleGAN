import os
import numpy as np

def normalize_Zscore(X_train, X_test):
    X=np.concatenate((X_train, X_test), axis=0)
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, X_test

def normalize_MinMax(X_train, X_test):
    X=np.concatenate((X_train, X_test), axis=0)
    X_min = np.min(X, axis = 0)
    X_max = np.max(X, axis = 0)
    X_train = (X_train - X_min) / (X_max-X_min)
    X_test = (X_test - X_min) / (X_max-X_min) 
    return X_train, X_test
    
def dense_to_one_hot(labels_dense, num_classes=4):
    return np.eye(np.max(labels_dense)+1)[labels_dense]

def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)
        
        
## Youngdo
        
def normalize_MeanVar(x):
    X_mean = np.mean(x, axis = 0) # by dimension
    X_var  = np.var (x, axis = 0)    
    X = np.nan_to_num((x - X_mean) / np.sqrt(X_var))
    # add 190523
    X_abs = np.max(abs(X), axis = 0)
    X = np.nan_to_num(X / X_abs)
    return X 

def normalize_MeanVar_train(x):
    X_mean = np.mean(x, axis = 0) # by dimension
    X_var = np.var(x, axis = 0)    
    X = np.nan_to_num((x - X_mean) / np.sqrt(X_var))
    # add 0523    
    X_abs = np.max(abs(X), axis = 0)
    X = np.nan_to_num(X / X_abs)
    return X, X_mean, X_var, X_abs

def normalize_MeanVar_by_train(x, X_mean, X_var, X_abs):   
    X = np.nan_to_num((x - X_mean) / np.sqrt(X_var))
    X = np.nan_to_num(X / X_abs)
    return X
def normalize_MeanVar_by_train_adt(x, X_mean, X_var):
    X = np.nan_to_num((x - X_mean) / np.sqrt(X_var))
    return X   
def unnormalize_abs_by_train(x, X_abs):   
    X = np.nan_to_num(x * X_abs)
    return X

def random_data(dataX, nsample=0):
    rand_x = np.arange(len(dataX))
    np.random.shuffle(rand_x)
    emos_X = []
    if nsample==0:
        nsample = len(dataX)
    for itn in range(nsample):
        tmp = dataX[rand_x[itn]]
        tmp = [tmp.tolist()]
        emos_X += tmp
    emos_X = np.array(emos_X)
    return emos_X