from sklearn.decomposition import PCA as sk_PCA
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
import pickle


# IMPLEMENTS THE PCA ALGORITHM
# CAN TAKE AS ARGUMENT THE NUMBER OF COMPONENTS TO KEEP
# THE reuse VARIABLE INDICATES IF THE MODEL HAS TO BE TRAINED
# AGAIN OR IF IT WILL JUST BE LOADED AND USED AS IS
# WARNING : MAKE SURE THAT A SAVED MODEL INSTANCE (.pkl) EXISTS
# THE TRAIN SET IS DOWNLOADED DIRECTLY IN THE FUNCTION BECAUSE OF ITS SIZE
def f_PCA(n_comp=30, reuse=False):
	# LOADING TRAIN SET, Y IS DOWNLOADED FOR PLOTTING PURPOSE
    X_train = np.load('Data_files/X_train_processed.npy', mmap_mode='r')
    Y_train = np.load('Data_files/Y_train.npy', mmap_mode='r')
    if reuse:
        with open('Algos/PCA_folder/PCA.pkl', 'rb') as filehandler:
            pca = pickle.load(filehandler)

    else:
		# CREATES PCA WITH n_comp COMPONENTS TO KEEP
		# AND WHITENS THE DATA (NORMALIZATION)
		# NOTE THAT PCA AUTOMATICALLY CENTERS THE DATA
        pca = sk_PCA(n_components=n_comp, whiten=True)
        X_train = np.load('Data_files/X_train_processed.npy', mmap_mode='r')
        pca.fit(X_train)
    # EXTRACTING THE VARIANCES AND THE EIGENSPECTRA FOR VISUALIZATION
    variances = pca.explained_variance_ratio_
    eig_vec = pca.components_
    
    # LOADING THE WAVELENGTHS FOR PLOTTING
    wavelengths = np.load('Data_files/wavelengths.npy', mmap_mode='r')
    fig = plt.figure() 
    # MEANINGFUL EIGENVECTORS/COMPONENTS SELECTED IN 1-BASED
    indexs_list = [8, 11, 20, 26, 30]
    for i in indexs_list:
        plt.plot(wavelengths, eig_vec[i-1], label='e'+str(i))
    plt.legend(fontsize=18)
    plt.grid()
    plt.xlabel('Wavelengths', fontsize=20)
    plt.ylabel('Intensity', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.savefig('Images/PCA_first_components.eps',
    #             format="eps", bbox_inches='tight')
    plt.show()
    del wavelengths

    # DICTIONARY CREATION TO ASSOCIATE TARGET TO COLOR
    dic = {}
    dic['0'] = 'b'
    dic['1'] = 'g'
    dic['2'] = 'r'
    dic['3'] = 'k'
    dic['4'] = 'c'
    coords = pca.transform(X_train)
    del X_train
    labels = ['QSO _', 'QSO BAL', 'GALAXY LRG', 'GALAXY ELG', 'STAR']
    
    # THE FOLLOWING FIGURES PLOT THE DISTRIBUTION OF LABELS 
    # ON THE SELECTED COMPONENTS
    fig = plt.figure()
    for i in np.unique(Y_train):
        plt.plot(coords[Y_train == i][19], coords[Y_train == i][25],
                 dic[str(int(i))]+'.', label=labels[int(i)],
                 markersize=7)
    plt.legend(fontsize=18)
    plt.grid()
    plt.xlabel('e20', fontsize=20)
    plt.ylabel('e26', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('Images/PCA_distribution2D_12.eps',
                format="eps", bbox_inches='tight')
    plt.show()

    fig = plt.figure()
    for i in np.unique(Y_train):
        plt.plot(coords[Y_train == i][19], coords[Y_train == i][29],
                 dic[str(int(i))]+'.', label=labels[int(i)],
                 markersize=7)
    plt.legend(fontsize=18)
    plt.grid()
    plt.xlabel('e20', fontsize=20)
    plt.ylabel('e30', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('Images/PCA_distribution2D_13.eps',
                format="eps", bbox_inches='tight')
    plt.show()

    fig = plt.figure()
    for i in np.unique(Y_train):
        plt.plot(coords[Y_train == i][25], coords[Y_train == i][29],
                 dic[str(int(i))]+'.', label=labels[int(i)],
                 markersize=7)
    plt.legend(fontsize=18)
    plt.grid()
    plt.xlabel('e26', fontsize=20)
    plt.ylabel('e30', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('Images/PCA_distribution2D_23.eps',
                format="eps", bbox_inches='tight')
    plt.show()

    # SAVES THE PCA OBJECT
    with open('Algos/PCA_folder/PCA.pkl', 'wb') as filehandler:
        pickle.dump(pca, filehandler)

    return pca, variances, eig_vec
