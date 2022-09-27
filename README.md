# Progetto Tirocinio 


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Denzel18/Tensorflow_Architecture_CODE">
    <img src="images/logo.jpg" alt="Logo" width="250" height="120">
  </a>
</div>
<br>

## About Project 
Project related to CV & DL course and master's thesis developed by Bernovschi D. and Giacomini A. Master's thesis
## Built With

* [Google Colab](https://colab.research.google.com/?hl=it)
* [Python 3.7.14](https://www.python.org/)

## Main Libraries 
* [TensorFlow](https://www.tensorflow.org) - v. 2.8.2
* [Keras](https://keras.io) - v. 2.8.0
* [Pytorch](https://pytorch.org)
* [FairLearn](https://fairlearn.org)
* [Scikit-Learn](https://scikit-learn.org/stable/)

## Files/Directory
* **OLD stuffs**: 
  * Pytorch Architectures
  * Custom Metrics 
  * Different Data Augmentation (D.A.) Libraries 
* **TENSORFLOW**
  * **TENSORFLOW - ADVERSARIAL** 
    * OLD VERSION (OLD ARCHITECTURE)
    * *Tensorflow Architecture Adversarial*: NO D.A., D.A. Offline & Online
  * **TENSORFLOW - BASELINE**
    * OLD VERSION 
    * *Tensorflow Architecture Base*: NO D.A., D.A. Offline & Online

* **Metrics**


## How it works - Tensorflow (Baseline and Adversarial architectures)
1. **Setting Parameters for Test**: Here we need to specify the type of IMAGE (CROP or NO_CROP) and the path of images 
2.  **Import**: Here is set the import of all libraries needed and the fix seed 
3.  **Drive**: Is the block for the connection to google drive storage 
4.  **Images Parameters**: Is the block where is set the path for the dataset (CSV) and the maps of Quality Class and Model Series
5.  **Split Data**: Here is defined the function to split the dataset and mantain the same ID stock in the same subset and "balance" the QC for each subset (val, train, test)
6.  **Data Generator**: For best idea of why is needed and what is the data generator, you can look here [LINK](https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3)
7.  **Network**: Is where is defined the newtork and specify the pretrained parameters. There is also an instruction to plot and save the model ```tf.keras.utils.plot_model(model, to_file=file_name_path, show_shapes=True)```
8.  **Metrics Balance Accuracy**: The implementation of Balance Accuracy 
9.  **Callbacks**: The callbacks is the implementations for early stopping model based on specified metrics. Model Checkpoint is the function to save the best model during the training 
10. **Preprocessing and Data Frame**: 
    1.  *check for leakage*: verify that same IDs are in the same sub-sets
    2.  ```class_weight.compute_class_weight```
    3.  ```split_data(...) ```
11. **D.A. OFFLINE**: D.A. Offline tecniques and Implementations how described in paper
12. **D.A. ONLINE**: D.A. Online tecniques and Implementations how described in paper
13. **Weighted Categorical Crossentropy Loss**
14. **Hyper Parameters**: Where sets the Optimizer, LR (Learning Rate), BS (Batch Size), DECAY, Momentum
15. **Creation of TrainGen, ValGen, TestGen**: The creations of custom Data Generator for train, test and validation dataframe. 
16. **Testing Model**: ```history = model.fit(x=traingen,validation_data=valgen, epochs=num_epochs, callbacks = [callbacks] , verbose=1)```
17. **Plot**: Plot the metrics 
18. **Save Model**: ```model.save(os.path.join(path+'weights/model_{}_{}/Final'.format(immgs,cnn)))```
19. **Load Model**: ```model = load_model(path_model, custom_objects={'balanced_accuracy_new':balanced_accuracy_new})```
20. **Prediction**: 
21. **Search univoque series to balance sets**: Here we split the prediction sets across model series consider
22. **Metrics (Mask & Img)**: The classification metrics like accuracy, precision, ... and the ordinal metrics (QWK, MAE, MS)
23. **Plot Confusion Matrix (C.M.) Function, Plot C.M.  General, Plot the C.M. for each model series**
24. **Cramer Correlation**: 
    1.  Calculate ***Cramers V*** statistic for categorial-categorial association. Uses correction from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013): 323-328
    2.  Alternative methods for the ***Cramers V*** 
25. **TSNE & PCA** (not used now)
    1.  ***(t-SNE) t-Distributed Stochastic Neighbor Embedding*** is a non-linear dimensionality reduction algorithm used for exploring high-dimensional data. 
    2.  ***(PCA) Principal Component Analysis*** is a technique used in multivariate statistics for simplification of source data. 
26. **New Metrics**:  Fairness Metrics and Balance Accuracy accross model series 

## How it works - Pytorch (GAN architecture)
1. sdfsdfds
2. 

### Paper

[LINK](https://www.google.it)

### Contributors 
- A. Giacomini 
- D. Bernovschi 

