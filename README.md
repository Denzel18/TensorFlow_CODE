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
    * OLD VERSION (OLD BASELINE ARCHITECTURE)
    * *Tensorflow Architecture Adversarial*: NO D.A., D.A. Offline & Online
  * **TENSORFLOW - BASELINE**
    * OLD VERSION (Other files with different CV transform libraries for D.A. - same architecture)
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
9.  **Callbacks**: The callbacks block is the implementations for early stopping model based on specified metrics. Model Checkpoint is the function to save the best model during the training 
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
20. **Prediction**: Performs the final predictions on the test sub-set of data.
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
1. **Import**: Here is set the import of all libraries needed and the fix seed 
2. **Custom TO CATEGORICAL**: 
3.  **Drive**: Is the block for the connection to google drive storage 
4.  **Images Parameters**: Is the block where is set the path for the dataset (CSV) and the maps of Quality Class and Model Series
5.  **Split Data**: Here is defined the function to split the dataset and mantain the same ID stock in the same subset and "balance" the QC for each subset (val, train, test)
6.  **Custom Dataset + Transforms D.A. techniques**: Composed by a custom DataLoader to perform the loading images phase on-the-fly, and another block with some transforms to perform even D.A. online with pytorch without using GAN architecture
7.  **Network**: Is where is defined the newtork and specify the pretrained parameters. There is also an instruction to plot and save the model ```tf.keras.utils.plot_model(model, to_file=file_name_path, show_shapes=True)```
8.  **Metrics Balance Accuracy**: The implementation of Balance Accuracy 
9.  **Early Stopping Class**: Imported class to monitor specific metrics during trainings to stop the training phase when the model stop to improve.
10. **Preprocessing and Data Frame**: 
    1.  *check for leakage*: verify that same IDs are in the same sub-sets
    2.  ```class_weight.compute_class_weight```
    3.  ```split_data(...) ```
11.  **Hyper Parameters**: Where sets the Optimizer, LR (Learning Rate), BS (Batch Size), DECAY, Momentum
12.  **Creation sub-sets (train,test,validation) with GAN created images**: contains 2 principal blocks:
  1. DA.offline - GAN images load from drive   
    1. to read all images name saved into the GAN architecture
    2. to concatenate the GAN images to the basic dataset
  2. Sets Extraction for Data Augmentation with GAN
    1. extract specific class-sets of data to be used into the GAN architecture exclusively 
    2. CustomDataset2 : to save in Drive folders of images for a specific class, in order to directly have the folder to be used in multiple GAN testings without the necessity of repeating the previous blocks of this step (point 12.) 
    3. Block to prepare the dataset to be used in the GAN architecture (THIS IS THE ONLY BLOCK NEEDED IF THE PATCHES FOLDERS ARE ALREADY CREATED)
13. **SAMPLER**: contains both strategies of "BalancedBatchSampler" & "ImbalancedBatchSampler"
14. **Creation of TrainGen, ValGen, TestGen**: The creations of custom Data Generator for train, test and validation dataframe. 
15. **GAN Evaluation Metrics - IMAGE QUALITY**: IMAGE QUALITY metrics to evaluate the goodness of the GAN architecture
16. **GAN Evaluation Metrics - IMAGE DIVERSITY**: IMAGE DIVERSITY metrics to evaluate the goodness of the GAN architecture
17. **Testing Model GAN**: blocks to perform the training of the GAN (both alternatives with and without OPTUNA hyperparameters fine-tuning)
18. **TRAIN VGG**: Block to alternative training the VGG16 architecture without executing the GAN, in order to evaluate in the same file the classificator with the images created with GAN.
19. **Plot**: Plot the metrics 
20. **Save Model**: ```model.save(os.path.join(path+'weights/model_{}_{}/Final'.format(immgs,cnn)))```
21. **Load Model**: ```model = load_model(path_model, custom_objects={'balanced_accuracy_new':balanced_accuracy_new})```
22. **Prediction**: Performs the final predictions on the test sub-set of data.
23. **Search univoque series to balance sets**: Here we split the prediction sets across model series consider
24. **Metrics (Mask & Img)**: The classification metrics like accuracy, precision, ... 
25. **Metriche secondarie** Ordinal metrics (QWK, MAE, MS)
26. **Plot Confusion Matrix (C.M.) Function, Plot C.M.  General, Plot the C.M. for each model series**
27. **Cramer Correlation**: 
    1.  Calculate ***Cramers V*** statistic for categorial-categorial association. Uses correction from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013): 323-328
    2.  Alternative methods for the ***Cramers V*** 
28. **TSNE & PCA** (not used now)
    1.  ***(t-SNE) t-Distributed Stochastic Neighbor Embedding*** is a non-linear dimensionality reduction algorithm used for exploring high-dimensional data. 
    2.  ***(PCA) Principal Component Analysis*** is a technique used in multivariate statistics for simplification of source data. 
29. **New Metrics**:  Fairness Metrics and Balance Accuracy accross model series 

### Contributors 
- A. Giacomini 
- D. Bernovschi 

