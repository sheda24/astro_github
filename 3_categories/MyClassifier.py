# -*- coding: utf-8 -*-
"""

THIS FILE IS THE TRAINING INTERFACE FOR THE MODELS

"""
# IMPORTING FILES AS NAMESPACES
import data_util as util
import Algos.PCA_folder.PCA as PCA
import Algos.SVC_folder.SVC as SVC
import Algos.GMM_folder.GMM as GMM
import Algos.RFC_folder.RFC as RFC
import Algos.NN_folder.NN1 as NN1
import Algos.NN_folder.NN2 as NN2
import Algos.KNN_folder.KNN as KNN
import Algos.STACK as STACK
import numpy as np
from numpy.lib.format import open_memmap

# SOURCE FILE USED FOR ALL SAMPLES
filename = r'/hpcstorage/raichoor/spplatelist_v5_11_0/' \
            'spall_redrock_v5_11_0.valid.fits'

# TRANSFERS THE DATA INTO .npy FILES TO REUSE
util.dl_filedata(filename, uncorrupted=True)

# GENERATES TRAIN SET AND TEST SET
print('Separating into training and testing sets')
util.generate_train_test_datasets()

# DUMMY PREPROCESSING FUNCTION IN CASE SPECIFIC PREPROCESSING IS IMPLEMENTED
print('Data preprocessing')
util.preprocessing('Data_files/X_train.npy')
util.preprocessing('Data_files/X_test.npy')

# PCA CREATION AND CALIBRATION BY THE TRAIN SET DISTRIBUTION
print('Applying PCA')
pca, variances, components = PCA.f_PCA(n_comp=30, reuse=True)

# APPLICATION OF PCA ON TRAIN DATASET
X_train = np.load('Data_files/X_train_processed.npy', mmap_mode='r')
X_train_final = open_memmap('Data_files/X_train_final.npy',
                            dtype='float32', mode='w+',
                            shape=(X_train.shape[0],pca.n_components_))

X_train_final[:] = pca.transform(X_train)
del X_train, X_train_final

# APPLICATION OF PCA ON TEST DATASET
X_test = np.load('Data_files/X_test_processed.npy', mmap_mode='r')
X_test_final = open_memmap('Data_files/X_test_final.npy',
                           dtype='float32', mode='w+',
                           shape=(X_test.shape[0],pca.n_components_))

X_test_final[:] = pca.transform(X_test)
del X_test, X_test_final

# LOADING DATASETS ONCE SMALL ENOUGH TO FIT IN MEMORY 
# THANKS TO PCA DIMENSIONALITY REDUCTION
print('Data loading')
X_train, Y_train, X_test, Y_test = util.load_train_test_datasets()

# TRAINING OF THE SUPPORT VECTOR CLASSIFIER
print('Training the support vector classification')
SVC_conf_mat = SVC.f_SVC(X_train, Y_train, X_test, Y_test,
                         reuse=True)

# TRAINING OF THE GAUSSIAN MIXTURE MODEL CLASSIFIER
print('Fitting with the gaussian mixture model')
GMM_conf_mat = GMM.f_GMM(X_train, Y_train, X_test, Y_test,
                         reuse=False)

# TRAINING OF THE RANDOM FOREST CLASSIFIER
print('Training the random forest classifier')
RFC_conf_mat = RFC.f_RFC(X_train, Y_train, X_test, Y_test,
                         reuse=True)

# TRAINING OF THE FULLY CONNECTED NEURAL NETWORK
print('Training neural network')
NN1_conf_mat = NN1.f_NN1(X_train, Y_train, X_test, Y_test,
                         reuse=True)

# TRAINING THE CONVOLUTIONAL NEURAL NETWORK
print('Training neural network')
NN2_conf_mat = NN2.f_NN2(X_train, Y_train, X_test, Y_test,
                         reuse=True)

# TRAINING THE K-NEAREST NEIGHBORS
print('Training KNN')
KNN_conf_mat = KNN.f_KNN(X_train, Y_train, X_test, Y_test,
                         reuse=True)

# TRAINS A STACKING ALGORITHM WITH PREVIOUSLY TRAINED MODELS
print('Now creating the stacked learning algorithm')
STACK_conf_mat = STACK.f_STACK(X_train, Y_train, X_test, Y_test ,
                               ['RFC','NN1'])

del X_train, Y_train, X_test, Y_test
