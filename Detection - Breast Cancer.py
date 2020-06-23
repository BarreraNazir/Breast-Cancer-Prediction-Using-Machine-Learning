#!/usr/bin/env python
# coding: utf-8

# # Detecting Breast Cancer

# Importing the Libraries

# In[5]:


import pandas as pd # Used to loading files
import numpy as np # Operate with mathematical operations
import matplotlib.pyplot as plt # Used for plotting
import seaborn as sns  # Used for visualizing


# Reading the dataset

# In[11]:


data = pd.read_csv("cancer.csv") # reading csv file


# In[12]:


data.head(5)


# Counting the distinct values in Diagnosis column

# In[14]:


data['diagnosis'].value_counts()


# Visualizing the Diagnosis Column

# In[18]:


f, ax = plt.subplots(1, 2, figsize = (15,5))
f.suptitle("Breat Cancer Patient", fontsize = 18.)
_ = data.diagnosis.value_counts().plot.bar(ax = ax[0], rot = 0, color = (sns.color_palette()[0], sns.color_palette()[2])).set(xticklabels = ["Yes", "No"])
_ = data.diagnosis.value_counts().plot.pie(labels = ("Malignant","Benign"), autopct = "%.2f%%", label = "", fontsize = 13., ax = ax[1],colors = (sns.color_palette()[0], sns.color_palette()[2]), wedgeprops = {"linewidth": 1.5, "edgecolor": "#F7F7F7"}), ax[1].texts[1].set_color("#F7F7F7"), ax[1].texts[3].set_color("#F7F7F7")


# Encoding the diagnosis column into categorical 

# In[22]:


from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
data.iloc[:,1] = labelencoder_Y.fit_transform(data.iloc[:,1].values)


# In[24]:


data.head(5)


# In[27]:


data.corr()


# In[36]:


X_Data = data.iloc[:,2:31].values
Y_Data = data.iloc[:,1].values


# Splitting data into training and testing 

# In[38]:


from sklearn import model_selection
# 0.3 means 30% will be used for testing and 0.7 or 70% data will be used for training
X_Train, X_Test, Y_Train, Y_Test = model_selection.train_test_split(X_Data, Y_Data, test_size = 0.3)


# In[39]:


print("Sample in training set...", X_Train.shape)
print("Sample in testing set...", X_Test.shape)
print("Sample in training set...", Y_Train.shape)
print("Sample in testing set...", Y_Test.shape)


# In[40]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[41]:


knnClassifier = KNeighborsClassifier()
knnClassifier.fit(X_Train, Y_Train)
Y_Pred = knnClassifier.predict(X_Test)
print(confusion_matrix(Y_Test, Y_Pred))
print(classification_report(Y_Test,Y_Pred))
accuracy = accuracy_score(Y_Test,Y_Pred)
print("Accuracy of KNN...", accuracy)


# In[42]:


dtreeClassifier = DecisionTreeClassifier()
dtreeClassifier.fit(X_Train, Y_Train)
Y_Pred = dtreeClassifier.predict(X_Test)
print(confusion_matrix(Y_Test, Y_Pred))
print(classification_report(Y_Test,Y_Pred))
accuracy = accuracy_score(Y_Test,Y_Pred)
print("Accuracy of Decision Tree...", accuracy)


# In[43]:


sgdClassifier = SGDClassifier()
sgdClassifier.fit(X_Train, Y_Train)
Y_Pred = sgdClassifier.predict(X_Test)
print(confusion_matrix(Y_Test, Y_Pred))
print(classification_report(Y_Test,Y_Pred))
accuracy = accuracy_score(Y_Test,Y_Pred)
print("Accuracy of SGD...", accuracy)


# In[44]:


svmClassifier = SVC(kernel='linear')
svmClassifier.fit(X_Train, Y_Train)
Y_Pred = svmClassifier.predict(X_Test)
print(confusion_matrix(Y_Test, Y_Pred))
print(classification_report(Y_Test,Y_Pred))
accuracy = accuracy_score(Y_Test,Y_Pred)
print("Accuracy of SVM...", accuracy)


# # Comparison of Machine Learning Classifiers

# The K Nearest Neighbor Classifier well performed among other classifiers. It means that given predictor will accurately guess the value of predicted attribute for a new data.

# In[45]:


Classifier = [['KNN', 95.3],['DT', 90.6 ],['SGD', 94.1 ],['SVM', 94.7] ]
result = pd.DataFrame(Classifier, columns = ['Classifier', 'Accuracy']) 
result.head()


# In[46]:


Accuracy = result['Accuracy'].values
Classifier = result['Classifier'].values
sns.set(style='darkgrid')
ax = result.plot(x="Classifier", y="Accuracy", kind="bar")
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
ax.get_legend().remove()
plt.show()

