import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import (Dense, Conv1D, MaxPooling1D, AveragePooling1D,
                          Dropout, Flatten, BatchNormalization)
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras import regularizers
from sklearn.metrics import confusion_matrix
import pickle


# VISUALIZATION FUNCTION CREATED TO SEE THE NETWORK'S HISTORY
# PLOTS THE LOSS FUNCTION FOR TRAIN SET AND VALIDATION SET FOR EACH EPOCH
# TRAINING AND VALIDATION ACCURACY FOR EACH EPOCH ON THE SECOND PLOT
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))
    ax1.set_ylabel('categorical cross entropy', fontsize=20)
    ax1.set_xlabel('epoch', fontsize=20)
    ax1.set_yscale('log')
    ax1.plot(history['loss'], label='training')
    ax1.plot(history['val_loss'], label='validation')
    ax1.legend(fontsize=18)
    ax1.tick_params(which='both', labelsize=18)
    ax1.grid()
    ax2.tick_params(which='both', labelsize=18)
    ax2.set_ylabel('accuracy [% correct]', fontsize=20)
    ax2.set_xlabel('epoch', fontsize=20)
    ax2.plot(history['acc'], label='training')
    ax2.plot(history['val_acc'], label='validation')
    ax2.legend(fontsize=18)
    ax2.grid()
    # plt.savefig('Images/Neural_Network2_perf.eps',
    #             format="eps", bbox_inches='tight')
    plt.show()


# THE reuse VARIABLE INDICATES IF THE MODEL HAS TO BE TRAINED
# AGAIN OR IF IT WILL JUST BE LOADED AND USED AS IS
# WARNING : MAKE SURE THAT A SAVED MODEL INSTANCE (.h5) EXISTS
def f_NN2(X_train, Y_train, X_test, Y_test, reuse):

    # LOADS THE SAMPLES UNPROCESSED
    # CNN ARE USED FOR VISUAL FEATURES DETECTION SO 
    # PROCESSING WOULD HARM THE APPEARANCE OF THE SPECTRA
    X_train = np.load('Data_files/X_train.npy', mmap_mode='r')
    X_test = np.load('Data_files/X_test.npy', mmap_mode='r')
    
    # CONVERTS THE TARGETS TO ONE HOT ENCODING VECTORS
    Y_train_ = keras.utils.to_categorical(Y_train)
    Y_test_ = keras.utils.to_categorical(Y_test)

    # RESHAPES THE INPUT TO BE 1 CHANNEL SAMPLES
    train_shape = (X_train.shape[0], X_train.shape[1], 1)
    test_shape = (X_test.shape[0], X_test.shape[1], 1)
    X_train_ = X_train.reshape(train_shape)
    X_test_ = X_test.reshape(test_shape)

    del X_train, X_test

    X_train_ = keras.utils.normalize(X_train_, axis=1)
    X_test_ = keras.utils.normalize(X_test_, axis=1)

    if reuse:
        path = 'Algos/NN_folder/NN2'
        with open(path+'_history.pkl', 'rb') as filehandler:
            history = pickle.load(filehandler)
        model = load_model(path+'.h5')
    else:
		# MODEL CONSTRUCTION
        model = Sequential()
        model.add(AveragePooling1D(pool_size=5))
        model.add(Conv1D(16, 3, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Conv1D(32, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='SGD',
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        call_back = ModelCheckpoint('Algos/NN_folder/weights.{epoch:02d}'
                                    '-{val_loss:.2f}.h5',
                                    monitor='val_loss')
        # FIT THE MODEL TO THE TRAINING DATA FOR 20 EPOCHS
        # BY BATCHES OF 30 SAMPLES 
        # VERBOSE IS THE DEGREE OF INFORMATIONS OUTPUT IN
        # THE TERMINAL DURING TRAINING
        # CALLBACKS (MODELCHECKPOINT HERE) SAVES THE INTERMEDIATE 
        # STATES OF THE NETWORK TO BE ABLE TO RESTART TRAINING 
        # FROM LAST STATE
        model_history = model.fit(X_train_, Y_train_,
                                  epochs=20, batch_size=30,
                                  validation_data=(X_test_, Y_test_),
                                  verbose=1, callbacks=[call_back])
        history = model_history.history
    # PLOTS THE MODEL HISTORY
    plot_history(history)

    # PREDICTS ON THE TEST SET AND COMPUTES THE CONFUSION MATRIX
    expected = np.argmax(model.predict(X_test_), axis=1)
    conf_matrix = confusion_matrix(expected, Y_test)
    normalisation = np.sum(conf_matrix, axis=1, keepdims=True)
    conf_matrix = conf_matrix/normalisation
    print(conf_matrix)

    # SAVES THE MODEL AND ITS HISTORY
    path = 'Algos/NN_folder/NN2'
    model.save(path+'.h5')
    # keras.utils.plot_model(model, to_file='Images/architecture_NN2.png')
    with open(path+'_history.pkl', 'wb') as filehandler:
        pickle.dump(history, filehandler)
    print('Model saved')

    return conf_matrix
