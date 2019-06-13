from sklearn.svm import LinearSVC as sk_SVC
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pickle


def f_SVC(X_train, Y_train, X_test, Y_test, reuse):

    if reuse:
        with open('Algos/SVC_folder/SVC.pkl', 'rb') as filehandler:
            classifier = pickle.load(filehandler)
    else:
        n_estim = 2
        classifier = sk_SVC(loss='hinge', class_weight='balanced',
                            verbose=2, max_iter=1000)
        print('Training the SVC')
        classifier.fit(X_train, Y_train)
    print('Now scoring the classification of the different labels')

    conf_matrix = confusion_matrix(classifier.predict(X_test), Y_test)
    normalisation = np.sum(conf_matrix, axis=1, keepdims=True)
    conf_matrix = conf_matrix/normalisation
    print(conf_matrix)

    with open('Algos/SVC_folder/SVC.pkl', 'wb') as filehandler:
        pickle.dump(classifier, filehandler)
    print('Model saved')

    return conf_matrix
