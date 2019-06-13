import astropy.io.fits as fits
import numpy as np
from numpy.lib.format import open_memmap
import os
import matplotlib.pyplot as plt
from random import shuffle


def dl_filedata(filename, uncorrupted, return_infos=False):
    # read zspec infos
    hdu = fits.open(filename)
    data = hdu[1].data
    hdu.close()
    # cut on reliable zspec and ELG targets
    print(len(data))
    # cut off SKY spectra and certain configurations
    if uncorrupted:
	    data = data[(data['ZWARN'] == 0) &
	                (data['OBJTYPE'] != 'SKY') &
	                (data['CHI2']/data['NPIXELS'].astype(float) > 0.4) &
	                (data['DELTACHI2']/data['NPIXELS'].astype(float) > 0.0025)]
	    print('ZWARN != 0, Sky and other factors taken out')
	    print(len(data))
    else:
        data = data[(data['ZWARN'] != 0)]
        print('Useless files taken out')
        print(len(data))
    pms = np.array([str(p)+'-'+str(m) for p, m in
                    zip(data['PLATE'], data['MJD']) if os.path.isfile(
                        r'/hpcstorage/raichoor/spplatelist_v5_11_0/'
                        'spPlate-'+str(p)+'-'+str(m)+'.fits')])
    print('List of fits file names created')
    # list of unique PLATE-MJD
    pmu = np.unique(pms)
    
    wavelengths_indexes = []
    end = len(pmu)
    # OPENS EACH SPPLATE-MJD FILE
    for i, pm in enumerate(pmu):
        hdu = fits.open(r'/hpcstorage/raichoor/spplatelist_v5_11_0/'
                        'spPlate-'+pm+'.fits')
        # COMPUTES THE WAVELENGTHS MEASURED                
        c0 = hdu[0].header['coeff0']
        c1 = hdu[0].header['coeff1']
        npix = hdu[0].header['naxis1']
        wave = 10.**(c0 + c1 * np.arange(npix))
        wave = np.around(wave, decimals=2)
        hdu.close()
        if i == 0:
			# IF FIRST FILE, SET THE WAVELENGTHS TO ITS wave
            wavelengths = wave
            wavelengths_indexes.append(np.arange(len(wavelengths)))
        else:
			# INTERSECT THE WAVELENGTHS OBSERVED ON OTHER FILES
			# TO THOSE FROM THE NEW ONE
			# GETS INDICES THAT CORRESPOND TO IDENTICAL WAVELENGTHS IN 
			# BOTH ARRAYS AND THEN NARROWS EACH OF THEM TO THE SHARED
			# WAVELENGTHS
            intersection, indexes_1, indexes_2 = np.intersect1d(
                                                        wavelengths,
                                                        wave,
                                                        return_indices=True)
			# wavelengths_indexes KEEPS TRACK OF THE INDICES 
			# TO KEEP FOR EACH FILE IT HAS ALREADY SEEN
			# AND ADDS THE INDICES OF THE NEW FILE
            wavelengths_indexes = [indexes[indexes_1]
                                   for indexes in wavelengths_indexes]
            wavelengths_indexes.append(indexes_2)
            # wavelengths HAS NOW TO BE UPDATED TO THE INTERSECTION
            # OF THE PREVIOUS SPECTRA WITH THE NEW ONE
            wavelengths = wavelengths[indexes_1]
        print('Loading wavelengths ' + str(i+1) + ' / ' + str(end))
    # SAVES THE COMMMON WAVELENGTHS 
    wl = open_memmap('Data_files/wavelengths.npy', dtype='float32',
                     mode='w+', shape=(len(wavelengths),))
    wl[:] = wavelengths
    del wl
    wavelengths_indexes = np.array(wavelengths_indexes)
    # GENERATES A LIST OF ALL THE FILENAMES 
    data_files = np.array([str(x)+'-'+str(y) for x, y
                           in zip(data['PLATE'], data['MJD'])])
    nb_fibers = len(pms)
    # CREATES A LIST OF LISTS CONTAINING THE INDEXES OF THE SELECTED 
    # SAMPLES FOR EACH FILE
    all_valid_ids = np.array([
                    np.array(data[data_files == filename]['FIBERID']-1)
                    for filename in pmu])
    # OUTPUTS THE SPPLATE-MJD AND FIBERIDS FOR DETECTION OR TRACABILITY
    if return_infos:
	    infos = open_memmap('Data_files/infos.npy', dtype='<U30',
	                        mode='w+', shape=(nb_fibers,2))
	    infos[:,0] = data_files
	    infos[:,1] = np.array([str(fib_id) for valid_ids in all_valid_ids for fib_id in valid_ids])
	    print(len(np.unique(infos[:,0])))  
	    del infos 
    del data_files, pms
    
    # GIVES NAME TO THE DATA FILES ACCORDING TO THE NATURE OF THE SAMPLES
    # IMPORTANT FOR CLASSIFICATION VS DETECTION
    if uncorrupted:
	    X_name = 'Data_files/X.npy'
	    Y_name = 'Data_files/Y.npy'
    else:
        X_name = 'Data_files/X_corrupted.npy'
        Y_name = 'Data_files/Y_corrupted.npy'

    X = open_memmap(X_name, dtype='float32', mode='w+',
                    shape=(nb_fibers, wavelengths_indexes.shape[1]))
    Y = open_memmap(Y_name, mode='w+', shape=(nb_fibers,))
    counter = 0
    for index, (pm, valid_ids) in enumerate(zip(pmu, all_valid_ids)):
        hdu = fits.open(r'/hpcstorage/raichoor/spplatelist_v5_11_0/'
                        'spPlate-'+pm+'.fits')
        print('Now uploading file'+pm+'.fits')
        # get the flux and ivar
        flux = np.array(hdu[0].data[valid_ids])[:, wavelengths_indexes[index]]
        X[counter:counter+len(valid_ids)][:] = np.array(flux)

        # ACQUIRES THE TYPES OF OBJECTS
        targets = [[x,y] for x,y in zip(data['SPECTYPE'][np.logical_and(
                         data['PLATE'] == hdu[0].header['PLATEID'],
                         data['MJD'] == hdu[0].header['MJD'])],
                   data['SUBTYPE'][np.logical_and(
                         data['PLATE'] == hdu[0].header['PLATEID'],
                         data['MJD'] == hdu[0].header['MJD'])])]
        # CONVERTS THE TYPES IN TARGETS FOR THE ALGORITHM
        LRG_list = ['BGS_0', 'LRG_1', 'BGS_2', 'LRG_3',
                    'ELG_4', 'BGS_17', 'ELG_19', 'BGS_20',
                    'ELG_26', 'BGS_28', 'ELG_30', 'BGS_39']
        targets_int = np.array([0 if (element[0] == 'QSO' and
                                      element[1].startswith('BAL'))
                                else 1 if (element[0] == 'QSO' and
                                           element[1].startswith('_'))
                                else 2 if (element[0] == 'GALAXY' and
                                           element[1] in LRG_list)
                                else 3 if (element[0] == 'GALAXY' and
                                           element[1] not in LRG_list)
                                else 4 for element in targets])
        Y[counter:counter+len(valid_ids)] = targets_int
        counter = counter+len(valid_ids)
        hdu.close()
    # PLOTS AN HISTOGRAM OF THE SAMPLES QUANTITY FOR EACH LABEL
    if uncorrupted:
        fig=plt.hist(Y, bins=np.arange(6), align='left')
        plt.xticks(np.arange(5),
                   labels=['QSO BAL_x', 'QSO _x', 'GALAXY LRG_x',
                           'GALAXY ELG_x', 'STAR'],
                   rotation=90, fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig('Images/Histogram_categories.eps',format="eps",bbox_inches='tight')
        plt.show()

    del X, Y

# DUMMY FUNCTION FOR PREPROCESSING
def preprocessing(filename):
    X = np.load(filename, mmap_mode='r')
    X_processed = open_memmap(filename[:-4] + '_processed.npy',
                                    dtype='float32', mode='w+',
                                    shape=X.shape)
    X_processed[:] = X
    del X, X_processed

# FUNCTION TO SEPARATE DATA INTO TRAIN AND TEST SET WITH A RATIO
def generate_train_test_datasets(ratio=0.7):
    Y = np.load('Data_files/Y.npy', mmap_mode='r')
	# LISTS THE DIFFERENT LABELS
    collection=np.unique(Y).astype(int)
    # TAKES SOME OF EACH LABEL WITH RELATIVE PROPORTION ratio
    # DUE TO THE COMPUTATION COST OF EXECUTING SHUFFLING, SLICING ETC
    # ON HIGH-DIMENSIONAL SAMPLES, INDEXES WERE PROCESSED INSTEAD OF THE
    # SAMPLES
    
    # COMPUTES THE NUMBER OF SAMPLES TO TAKE FROM EACH CATEGORY
    minimum = np.array([int(ratio*len(Y[Y == i])) for i in collection])
    indexes = np.arange(len(Y))
    
    # CREATES THE LIST OF THE INDEXES CORRESPONDING TO EACH CLASS
    category_indexes = [indexes[Y == i] for i in collection]
    
    # SHUFFLES THE INDEXES FOR EACH CLASS TO TAKE UNIFORMLY AT RANDOM 
    # minimum[i] SAMPLES FROM THE CLASS DESIGNATED BY THIS INDEX i
    for i in collection:
	    shuffle(category_indexes[i])

    indexes_train_interm = [category_indexes[i][:minimum[i]] for i in collection]
    indexes_train = np.array([index for category in indexes_train_interm
                           for index in category])
    
    # AFTER FUSING ALL THE TRAIN SAMPLES FROM EACH LABEL TOGETHER,
    # THEY ARE SHUFFLED FOR THE ALGORITHMS
    shuffle(indexes_train)
    X = np.load('Data_files/X.npy', mmap_mode='r')
    X_train = open_memmap('Data_files/X_train.npy', dtype='float32',
                          mode='w+',
                          shape=(np.sum(minimum), X.shape[1]))

    Y_train = open_memmap('Data_files/Y_train.npy', mode='w+',
                          shape=(np.sum(minimum),))
                         
    # THE SAMPLES ARE FINALLY EXTRACTED FROM THE TRAIN SET
    X_train[:] = X[indexes_train]                      
    Y_train[:] = Y[indexes_train]

    del X_train, Y_train
    
    # THE REMAINING SAMPLES NOT EXTRACTED FOR THE TRAINING FOLLOW
    # THE SAME PROCESS TO MERGE INTO THE TESTING SET
    X_test = open_memmap('Data_files/X_test.npy', dtype='float32',
                         mode='w+',
                         shape=(X.shape[0]-np.sum(minimum), X.shape[1]))
    
    Y_test = open_memmap('Data_files/Y_test.npy', mode='w+',
                         shape=(X.shape[0]-np.sum(minimum),))

    indexes_test_interm = [category_indexes[i][minimum[i]:] for i in collection]
    indexes_test = np.array([index for category in indexes_test_interm
                          for index in category])
    shuffle(indexes_test)
    X_test[:] = X[indexes_test]
    Y_test[:] = Y[indexes_test]

                         
    del X, Y, X_test, Y_test


def load_train_test_datasets():
	# LOADS THE DATA WHEN LIGHT IN VOLUME
	# PCA REDUCED DIMENSION FROM 4000 TO 30
    X_train = np.load('Data_files/X_train_final.npy', mmap_mode='r')
    Y_train = np.load('Data_files/Y_train.npy', mmap_mode='r')
    X_test = np.load('Data_files/X_test_final.npy', mmap_mode='r')
    Y_test = np.load('Data_files/Y_test.npy', mmap_mode='r')

    return X_train, Y_train, X_test, Y_test
