Project folder structure:
-Projet/
  -3_categories/
    -MyClassifier.py
    -detection.py
    -data_util.py
    -Data_files/
      Various data files from the original data to the transformed through PCA, all are .npy files (better than pkl for numpy arrays:
      for memory size, writing and loading time)
    -Algos/
      Folders containing the .py file of the model and the saved model object (.pkl for classic ml models and .h5 for deep learning models), the only
      folder containing multiple algorithms is the Neural Network folder (NN_folder)
      STACK.py 
      STACK.pkl was not created since its result were unsatisfying, but it can be created by executing STACK.py 
    -Images/
      Images used for the report
  -5_categories/
    -MyClassifier.py
    -detection.py
    -data_util.py
    -Data_files/
      Various data files from the original data to the transformed through PCA, all are .npy files (better than pkl for numpy arrays:
      for memory size, writing and loading time)
    -Algos/
      Folders containing the .py file of the model and the saved model object (.pkl for classic ml models and .h5 for deep learning models), the only
      folder containing multiple algorithms is the Neural Network folder (NN_folder)
      STACK.py 
      STACK.pkl
    -Images/
      Images used for the report

The main characteristics apart from the structure are the use of Keras for the Deep Learning framework, maybe a bit less comfortable for research
(compared to PyTorch) but really efficient for creating relatively simple architectures and leverage the use of GPU nodes. The use of numpy memory
maps to deal with the size of the data is also something worth mentioning. The implementation is quite straightforward and just requires some
precision during the declaration of the variable. Its mechanisms do not need to be fully understood to be handled. The functions used to prepare
the data for the algorithms are all in the data_utils.py . Although the architecture separates the classification between 3 and 5 categories,
it is mainly done so that the models saved and the data containing the labels can be handled easily. A fused architecture could be envisioned at the 
cost of global variables or tests each time the number of categories matter.
