import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import (Dense, Conv1D, MaxPooling1D,
                          Dropout, Flatten, BatchNormalization)
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
    # plt.savefig('Images/Neural_Network1_perf.eps',
    #             format="eps", bbox_inches='tight')
    plt.show()


# THE reuse VARIABLE INDICATES IF THE MODEL HAS TO BE TRAINED
# AGAIN OR IF IT WILL JUST BE LOADED AND USED AS IS
# WARNING : MAKE SURE THAT A SAVED MODEL INSTANCE (.h5) EXISTS
def f_NN1(X_train, Y_train, X_test, Y_test, reuse):
    
    # CONVERTS THE TARGETS TO ONE HOT ENCODING VECTORS 
    Y_train_ = keras.utils.to_categorical(Y_train)
    Y_test_ = keras.utils.to_categorical(Y_test)
    print(Y_train.shape)
    print(Y_test.shape)

    if reuse:
        path = 'Algos/NN_folder/NN1'
        with open(path+'_history.pkl', 'rb') as filehandler:
            history = pickle.load(filehandler)
        model = load_model(path+'.h5')
    else:
		# MODEL CONSTRUCTION
        model = Sequential()
        model.add(Dense(90, activation='relu'))
        model.add(Dense(45, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        model.compile(optimizer='SGD',
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        # FIT THE MODEL TO THE TRAINING DATA FOR 200 EPOCHS
        # BY BATCHES OF 30 SAMPLES 
        # VERBOSE IS THE DEGREE OF INFORMATIONS OUTPUT IN
        # THE TERMINAL DURING TRAINING
        model_history = model.fit(X_train, Y_train_,
                                  epochs=200, batch_size=30,
                                  validation_data=(X_test, Y_test_),
                                  verbose=2)
        history = model_history.history
    # PLOTS THE MODEL HISTORY
    plot_history(history)

    # PREDICTS ON THE TEST SET AND COMPUTES THE CONFUSION MATRIX
    expected = np.argmax(model.predict(X_test), axis=1)
    conf_matrix = confusion_matrix(expected, Y_test)
    normalisation = np.sum(conf_matrix, axis=1, keepdims=True)
    conf_matrix = conf_matrix/normalisation
    print(conf_matrix)

    # SAVES THE MODEL AND ITS HISTORY
    path = 'Algos/NN_folder/NN1'
    model.save(path+'.h5')
    # keras.utils.plot_model(model, to_file='architecture_NN1.png')
    with open(path+'_history.pkl', 'wb') as filehandler:
        pickle.dump(history, filehandler)
    print('Model saved')

    return conf_matrix
