from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

# IMPLEMENTS THE K-NEAREST NEIGHBORS ALGORITHM
# TAKES AS INPUT THE TRAIN SET FOR TRAINING
# AND THE TEST SET FOR VALIDATION
# THE reuse VARIABLE INDICATES IF THE MODEL HAS TO BE TRAINED
# AGAIN OR IF IT WILL JUST BE LOADED AND USED AS IS
# WARNING : MAKE SURE THAT A SAVED MODEL INSTANCE (.pkl) EXISTS
def f_KNN(X_train, Y_train, X_test, Y_test, reuse):
    if reuse:
        with open('Algos/KNN_folder/KNN.pkl', 'rb') as filehandler:
            classifier = pickle.load(filehandler)
    else:
		# CREATES A KNN WITH COMPARISON WITH THE 50 NEAREST NEIGHBORS
		# n_jobs IS THE NUMBER OF PARALLEL COMPUTATIONS TO ALLOW
		# -1 BEING THE MAXIMUM POSSIBLE
        classifier = KNN(n_neighbors=50, n_jobs=-1)
        print('Training the KNN')
        # TRAINING THE KNN WITH THE TRAIN SET
        classifier.fit(X_train, Y_train)
    print('Now scoring the classification of the different labels')

    # EVALUATION ON THE TEST SET AND CONFUSION MATRIX CREATION
    # WARNING : FOR KNN THE PREDICTION IS REALLY LONG AS IT HAS
    # TO COMPUTE THE DISTANCE TO EACH SAMPLE 
    conf_matrix = confusion_matrix(classifier.predict(X_test), Y_test)
    normalisation = np.sum(conf_matrix, axis=1, keepdims=True)
    conf_matrix = conf_matrix/normalisation
    print(conf_matrix)

    # SAVES THE MODEL
    with open('Algos/KNN_folder/KNN.pkl', 'wb') as filehandler:
        pickle.dump(classifier, filehandler)
    print('Model saved')
    
    return conf_matrix
