from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

# IMPLEMENTS THE RANDOM FOREST CLASSIFIER
# IT TAKES AS INPUT THE TRAIN SET AND LABELS 
# FOR CALIBRATION AND THE TEST SET AND LABELS
# FOR EVALUATION AND CONFUSION MATRIX
# THE reuse VARIABLE INDICATES IF THE MODEL HAS TO BE TRAINED
# AGAIN OR IF IT WILL JUST BE LOADED AND USED AS IS
# WARNING : MAKE SURE THAT A SAVED MODEL INSTANCE (.pkl) EXISTS
def f_RFC(X_train, Y_train, X_test, Y_test, reuse):
    if reuse:
        with open('Algos/RFC_folder/RFC.pkl', 'rb') as filehandler:
            classifier = pickle.load(filehandler)
    else:
		# n_estim IS THE NUMBER OF TREES TO TRAIN SEPARATELY
		# BEFORE BEING COMPILED TOGETHER AS A RANDOM FOREST	
		# THE max_depth IS THE MAXIMUM NUMBER OF DECISIONS/NODES
		# IN THE DECISION TREE
		# n_jobs IS THE NUMBER OF PARALLEL COMPUTATIONS TO ALLOW
		# -1 BEING THE MAXIMUM POSSIBLE	
        n_estim = 50
        classifier = RandomForestClassifier(n_estimators=n_estim, max_depth=15,
                                            n_jobs=20, class_weight='balanced')
        # TRAINING THE RFC WITH THE TRAIN SET        
        print('Training the RandomForest')
        classifier.fit(X_train, Y_train)
    print('Now scoring the classification of the different labels')

    # SCORING THE MODEL ON THE TEST SET GIVEN IN ARGUMENT
    # AND GENERATING THE CONFUSION MATRIX
    conf_matrix = confusion_matrix(classifier.predict(X_test), Y_test)
    normalisation = np.sum(conf_matrix, axis=1, keepdims=True)
    conf_matrix = conf_matrix/normalisation
    print(conf_matrix)

    # SAVES THE MODEL
    with open('Algos/RFC_folder/RFC.pkl', 'wb') as filehandler:
        pickle.dump(classifier, filehandler)
    print('Model saved')
    
    return conf_matrix
