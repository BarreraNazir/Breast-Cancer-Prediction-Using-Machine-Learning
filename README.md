# Breast Cancer Detector
 Predicting the type of breast cancer (Malignant or Benign)

Overview

Breast cancer is the most common form of cancer among women and the second most common cancer in the world. Survival is directly related to stage at diagnosis, as can be seen by a 98% 10-year survival rate for patients with stages 0 and I disease compared with a 65% 10-year survival rate for patients with stage III disease. To improve survival in this disease, more patients need to be identified at an early stage. Therefore, we evaluated existing and emerging technologies used for breast cancer screening and detection to identify areas for potential improvement. 

This task consist of three phases

Breast Cancer Dataset:

1 Dataset overview:

     The breast cancer dataset is a classic and binary classification. This data set includes 201 instances of one class and 85 instances of another class. The instances are described by 9 attributes, some of which are linear and some are nominal.

2 Pre-processing:

     The dataset is entirely clean, no corrupt data. In this step, the class lebel 'diagnosis' is converted into categorical with the help of LabelEncoder.

3 Classification:

     It predict numeric labels, to categorize patient as either malignant or benign. The classification include two steps:
 
    - Building the classifier or model:
 
         The classifier is build from the training set including the data of dependent and independent variable. 
     
    - Using classifier for classification:
 
         The classifier is used for classification. Here the test data is used to estimate the accuracy of models. The classifier can be applied to new data if results is      considered as acceptable. The four classifier models were used to diagnose the patient, namely KNN, DT, SGD, and SVM. Later on, the results of all classifiers were compared to generate 100 % results. 
     

