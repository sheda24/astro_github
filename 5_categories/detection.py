# -*- coding: utf-8 -*-
"""

THIS FILE IS FOR THE DOUBLE-SPECTRA DETECTION

"""
import data_util as util
import numpy as np
import pickle
from numpy.lib.format import open_memmap
from keras.models import load_model
import keras
import matplotlib.pyplot as plt

# CREATING THE DIFFERENTS LISTS : 
#      - THE LIST OF ALGORITHMS TO BE USED
#      - THE DOUBLE SPECTRA LIST, A LIST OF LISTS,
#        EACH BEING THE SELECTED INDEXES BY HAND 
#        FOR EACH ALGORITHM
algorithms = ['NN1', 'NN2']
indexes_selected = [[15917, 15477, 15227, 15231, 15045],
                    [47731, 47560, 47538, 47532, 47265]]

# SOURCE FILE USED FOR ALL SAMPLES
filename = r'/hpcstorage/raichoor/spplatelist_v5_11_0/' \
            'spall_redrock_v5_11_0.valid.fits'

# TRANSFERS THE DATA INTO .npy FILES TO REUSE
util.dl_filedata(filename, uncorrupted=False, return_infos=True)

# DUMMY PREPROCESSING FUNCTION IN CASE SPECIFIC PREPROCESSING IS IMPLEMENTED
util.preprocessing('Data_files/X_corrupted.npy')

# LOADING THE PCA CALIBRATED IN MyClassifier.py
with open('Algos/PCA_folder/PCA.pkl', 'rb') as filehandler:
    pca = pickle.load(filehandler)

# APPLICATION OF PCA ON DATASET
X_corrupted = np.load('Data_files/X_corrupted_processed.npy')
X_corrupted_final = open_memmap('Data_files/X_corrupted_final.npy',
                                dtype='float32', mode='w+',
                                shape=(X_corrupted.shape[0],
                                       pca.n_components_))
X_corrupted_final[:] = pca.transform(X_corrupted)

# LOADING THE WAVELENGTHS OF THE SPECTRA FOR PLOTTING PURPOSE
wavelengths = np.load('Data_files/wavelengths.npy', mmap_mode='r')

# LOADING THE DATA ONCE IT FITS IN MEMORY, AND ITS INFORMATIONS
# TO ALLOW SPPLATE-MJD AND FIBERID TRACING
# dataset IS THE ORIGINAL DATA WITHOUT ANY PROCESSING
X_corrupted_final = np.load('Data_files/X_corrupted_final.npy', mmap_mode='r')
X_corrupted_infos = np.load('Data_files/infos.npy', mmap_mode='r')
dataset = np.load('Data_files/X_corrupted.npy', mmap_mode='r')

# LOADING THE MODELS SPECIFIED IN THE LIST
# AND FINDS THE ELEMENTS SPECIFIED AS SUSPECT BY THIS ALGORITHM
# THE INDEXES ARE SPECIFIED FOR PLOTTING PURPOSE AND THE LIST
# CAN BE EMPTY BEFORE SELECTING THE CANDIDATES FOR DOUBLE-SPECTRA
for liste, algorithm in zip(indexes_selected, algorithms):

	# VARIOUS CONDITIONS APPLY FOR THE LOADING AND THE 
	# PREDICTION DEPENDING ON THE MODEL'S NATURE OR
	# SAVED FORMAT
    if algorithm is 'STACK':
        with open('Algos/'+algorithm+'.pkl', 'rb') as filehandler:
            model = pickle.load(filehandler)
            results = model.predict_proba(X_corrupted_final)
    else:
        if algorithm.startswith('NN'):
            path = 'Algos/NN_folder/'+algorithm
            model = load_model(path+'.h5')
            if algorithm is 'NN2':
                dataset_shape = (dataset.shape[0], dataset.shape[1], 1)
                dataset_ = dataset.reshape(dataset_shape)
                dataset_ = keras.utils.normalize(dataset_, axis=1)
                results = model.predict(dataset_)
                # CONVERTS THE NETWORK OUTPUT TO PROBABILITIES
                results = np.exp(results)/np.sum(np.exp(results), axis=1, keepdims=True)
                del dataset_
            else:
                results = model.predict(X_corrupted_final)
        else:
            object_path = 'Algos/'+algorithm+'_folder/'+algo+'.pkl'
            with open(object_path, 'rb') as filehandler:
                algorithms.append(pickle.load(filehandler))
                results = model.predict_proba(X_corrupted_final)

    # SORTS THE PROBABILITIES PER SAMPLE BY MAGNITUDE
    # NOT BY CLASS LABEL 
    probabilities = np.sort(results)
    
    # SORTS ACCORDING TO THE HIGHEST PROBABILITY
    indexes_increasing = np.argsort(probabilities[:, -1])
    probabilities = probabilities[indexes_increasing]
    infos = X_corrupted_infos[indexes_increasing]

    # COMPUTATION OF THE PROBABILITY DIFFERENCES FOR PLOTTING
    diff12 = probabilities[:, -1]-probabilities[:, -2]
    diff13 = probabilities[:, -1]-probabilities[:, -3]
    diff23 = probabilities[:, -2]-probabilities[:, -3]

    # PRINTING THE CANDIDATE SPECTRA INFORMATIONS 
    # ALONG WITH THE PROBABILITIES COMPUTED
    for index in liste:
        probas_original = results[indexes_increasing[index]]
        print('PLATE-MJD: '+infos[index][0]+'   FIBER ID: '+infos[index][1])
        print('The probabilities found were:'
              + '  QUASAR BAL('+str(probas_original[0])
              + ')    -    QUASAR _ ('+str(probas_original[1])
              + ')    -    GALAXY LRG('+str(probas_original[2])+')'
              + ')    -    GALAXY ELG('+str(probas_original[3])+')'
              + ')    -    STAR('+str(probas_original[4])+')')
              
    # PLOTTING THE CANDIDATES' SPECTRE
    for i in liste:
        fig = plt.figure()
        plt.xlabel('Wavelengths', fontsize=20)
        plt.ylabel('Intensity', fontsize=20)
        plt.plot(wavelengths, dataset[indexes_increasing[i]])
        plt.grid()
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()
    # PLOT OF THE PROBABILITIES DISTRIBUTION TO GET INSIGHT
    fig = plt.figure()
    for i in range(probabilities.shape[1]):
        if i == 0:
            plt.plot(probabilities[:, ::-1][:, i],
                     label=str(i+1)+'st probability')
        elif i == 1:
            plt.plot(probabilities[:, ::-1][:, i],
                     label=str(i+1)+'nd probability')
        elif i == 2:
            plt.plot(probabilities[:, ::-1][:, i],
                     label=str(i+1)+'rd probability')
        else:
            plt.plot(probabilities[:, ::-1][:, i],
                     label=str(i+1)+'th probability')
    plt.xlabel('Spectra sorted by certainty', fontsize=20)
    plt.ylabel('Probability by model', fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    # plt.savefig('Images/Probabilities_corrupted_spectra_'+algorithm+'.eps',format="eps",bbox_inches='tight')
    plt.show()

    # PLOT OF THE DIFFERENCE BETWEEN THE TWO MAJOR PROBABILITIES
    # OF OUTPUT, USED TO SELECT CANDIDATES
    fig = plt.figure()
    plt.plot(diff12, '.')
    plt.xlabel('Spectra sorted by certainty', fontsize=20)
    plt.ylabel(r'$p_{maj}-p_{nd \, maj}$', fontsize=20)
    plt.grid()
    plt.show()

    # PLOT OF THE DIFFERENCE SBETWEEN THE THREE MOST PROBABLE OUTPUTS
    # USEFUL TO ESTIMATE THE NETWORK CERTAINTY
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 10))
    ax1.plot(diff12, '.')
    ax2.plot(diff13, '.')
    ax3.plot(diff23, '.')
    ax1.set_xlabel('Spectra sorted by certainty', fontsize=20)
    ax1.set_ylabel(r'$p_{maj}-p_{nd \, maj}$', fontsize=20)
    ax1.tick_params(which='both', labelsize=18)
    ax2.set_xlabel('Spectra sorted by certainty', fontsize=20)
    ax2.set_ylabel(r'$p_{maj}-p_{rd \, maj}$', fontsize=20)
    ax2.tick_params(which='both', labelsize=18)
    ax3.set_xlabel('Spectra sorted by certainty', fontsize=20)
    ax3.set_ylabel(r'$p_{nd \, maj}-p_{rd \, maj}$', fontsize=20)
    ax3.tick_params(which='both', labelsize=18)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()

del X_corrupted, X_corrupted_final, X_corrupted_infos, dataset
