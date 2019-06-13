from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy.lib.format import open_memmap
import pickle


# IMPLEMENTS THE STACKING ALGORITHM
# TAKING AS INPUT A LIST OF MODELS TO USE TOGETHER
# THE reuse VARIABLE INDICATES IF THE MODEL HAS TO BE TRAINED
# AGAIN OR IF IT WILL JUST BE LOADED AND USED AS IS
# WARNING : MAKE SURE THAT A SAVED MODEL INSTANCE (.pkl) EXISTS
def f_STACK(X_train, Y_train, X_test, Y_test, models, reuse=False):
    if reuse:
        with open('Algos/STACK.pkl', 'rb') as filehandler:
            classifier = pickle.load(filehandler)
    # THE STACKING ALGORITHM IS A LOGISTIC CLASSIFIER
    else:
        classifier = LogisticRegression(multi_class='multinomial',
                                        solver='saga', verbose=0,
                                        class_weight='balanced', n_jobs=20)
        algorithms_predictions = np.empty(
                                 shape=(X_train.shape[0], len(models))
                                 )

        # LOADS THE MODEL AND SAVES THEIR PREDICTIONS
        for i, algo in enumerate(models):
            if algo.startswith('NN'):
                path = 'Algos/'+algo[:-1]+'_folder/'+algo+'.pkl'
                with open(path, 'rb') as filehandler:
                    model, model_history = pickle.load(filehandler)
                algorithms_predictions[:, i] = np.argmax(
                                               model.predict(X_train),
                                               axis=1)
            else:
                path = 'Algos/'+algo+'_folder/'+algo+'.pkl'
                with open(path, 'rb') as filehandler:
                    model = pickle.load(filehandler)
                algorithms_predictions[:, i] = model.predict(X_train)
        # TRAINS THE LOGISTIC CLASSIFIER TO CHOOSE OUTPUT DEPENDING
        # ON MODELS' DECISION
        classifier.fit(algorithms_predictions, Y_train)

    # NOW APPLIES ITSELF ON THE TEST SET, IT MUST FIRST OBTAIN THE 
    # PREDICTIONS OF THE SUBMODELS
    algorithms_tests = np.empty(shape=(X_test.shape[0], len(models)))
    for i, algo in enumerate(models):
        if algo.startswith('NN'):
            path = 'Algos/'+algo[:-1]+'_folder/'+algo+'.pkl'
            with open(path, 'rb') as filehandler:
                model, model_history = pickle.load(filehandler)
            algorithms_tests[:, i] = np.argmax(
                                     model.predict(X_test),
                                     axis=1)
        else:
            path = 'Algos/'+algo+'_folder/'+algo+'.pkl'
            with open(path, 'rb') as filehandler:
                model = pickle.load(filehandler)
            algorithms_tests[:, i] = model.predict(X_test)
    expected = classifier.predict(algorithms_tests)

    # COMPUTES THE CONFUSION MATRIX
    conf_matrix = confusion_matrix(expected, Y_test)
    normalisation = np.sum(conf_matrix, axis=1, keepdims=True)
    conf_matrix = conf_matrix/normalisation
    print(conf_matrix)
    
    # SAVES THE MODEL
    with open('Algos/STACK.pkl', 'wb') as filehandler:
        pickle.dump(classifier, filehandler)
    print('Model saved')

    return conf_matrix
