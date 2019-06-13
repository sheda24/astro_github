from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture as sk_GMM
import numpy as np
import pickle


def f_GMM(X_train, Y_train, X_test, Y_test, reuse):

    if reuse:
        with open('Algos/GMM_folder/GMM.pkl', 'rb') as filehandler:
            classifier = pickle.dump(filehandler)
    else:
        classifier = sk_GMM(n_components=9, n_init=10)
        print('Training the GMM')
        classifier.fit(X_train, Y_train)

    print('Now scoring the classification of the different labels')
    conf_matrix = confusion_matrix(classifier.predict(X_test), Y_test)
    normalisation = np.sum(conf_matrix, axis=1, keepdims=True)
    conf_matrix = conf_matrix/normalisation
    print(conf_matrix)

    with open('Algos/GMM_folder/GMM.pkl', 'wb') as filehandler:
        pickle.dump(classifier, filehandler)

    return conf_matrix
