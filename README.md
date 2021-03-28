# Breast Cancer Prediction Using Machine Learning

**Problem Identification**:

Breast cancer is the most common malignancy among women, accounting for nearly 1 in 3 cancers diagnosed among women in the United States, and it is the second leading cause of cancer death among women. Breast Cancer occurs as a results of abnormal growth of cells in the breast tissue, commonly referred to as a Tumor. A tumor does not mean cancer - tumors can be benign (not cancerous), pre-malignant (pre-cancerous), or malignant (cancerous). Tests such as MRI, mammogram, ultrasound and biopsy are commonly used to diagnose breast cancer performed.

**Goal**:

The goal is to classify whether the breast cancer patient is diagnosed as benign or malignant. To achieve this we have used machine learning classification methods to fit a function that can predict the class of new patient.

**Getting Started**

The project is broken down into the following steps:

* Importing Libraries and Data
* Data Pre-Processing
* Data Visualization
* Model Building


**Importing Libraries and Data**

  * Importing Libraries:
     
   Used Pandas, NumPy, Matplotlib, Seaborn and Sklearn Libraries 
     
   * Dataset:
     
   The Breast Cancer datasets is available machine learning repository maintained by the University of California, Irvine. The dataset contains 569 samples of malignant and benign tumor cells. The first two columns in the dataset store the unique ID numbers of the samples and the corresponding diagnosis (M=malignant, B=benign), respectively. The columns 3-32 contain 30 real-value features that have been computed from digitized images of the cell nuclei, which can be used to build a model to predict whether a tumor is benign or malignant.

**Data Pre-Processing:**  

   Data preprocessing is a crucial step for any data analysis problem. It is often a very good idea to prepare your data in such way to best expose the structure of the problem to the machine learning algorithms that you intend to use. Here in this model it involves:

   * Label Encoder
   
   After encoding the class labels(diagnosis) in an array, the malignant tumors are now represented as class 1 (i.e prescence of cancer cells) and the benign tumors are represented as class 0 (i.e no cancer cells detection), respectively, illustrated by calling the transform method of LabelEncorder on two dummy variables
     
**Data Visualization:**

   It is essential to have an overall picture of data Basic statistical descriptions used to identify properties of the data and highlight which data values should be treated as noise or outliers. Here two approaches used to explore the data:
   
   * Descriptive statistics:
          
   It's the process of condensing key characteristics of the dataset into simple numeric metrics. Some of the common metrics used are mean, symmetry and correlation.
   
   * Visualization:
       
   It is the process of projecting the data, or parts of it, into Cartesian space or into abstract images. In the data mining process, data exploration is leveraged in many different steps including preprocessing, modeling, and interpretation of results.
          
**Model Building:** 
   
   The simplest method to evaluate the performance of a machine learning algorithm is to use different training and testing datasets. Here I split the available data into a training set and a testing set (70% training, 30% test).Train the algorithm on the first part, then make predictions on the another part, and then compare the classifier results namely K-NN, Decision Tree, Logistic Regression, and Support Vector Machine. Among all, K-NN provide better accuracy results. The size of the split can depend on the size and specifics of your dataset, although it is common to use 67% of the data for training and the remaining 33% for testing. 
