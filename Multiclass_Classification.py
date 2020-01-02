#!/usr/bin/env python
# coding: utf-8

# # Multiclass Classification
# Now let’s try to detect more than just the 5s. In order to do so, we are required to import differnt libraries as well as dataset here

# In[1]:


## Importing dataset from fetch_openml
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version= 1)
mnist.keys()


# In[2]:


## checking the sape of the datasets
X, y = mnist['data'], mnist['target']
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# In[3]:


## Instance’s feature vector, reshaping it to a 28 × 28 array, and displaying it using Matplotlib’s imshow() function:
import matplotlib as mlb
import matplotlib.pyplot as plt

some_digit = X[12]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap = "binary")
##plt.axis("off")
plt.show()


# In[4]:


## Splitting the training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
print("X_train, X_test:", X_train.shape, X_test.shape)
print("y_train, y_test:", y_train.shape, y_test.shape)


# Above, we have imported dataset, importatant libraries, splitted data into training and testing dataset

# Scikit-Learn detects when you try to use a binary classification algorithm (SVM) for a multiclass classification task, and it automatically runs OvR (one-versus-rest) or OvO (one-versus-one), depending on the algorithm. Let’s try this with a Support Vector Machine classifier, using the sklearn.svm.SVC class:
# 

# In[5]:


from sklearn.svm import SVC


# In[6]:


svm_clf = SVC(gamma="auto", random_state=42)


# In[7]:


## Due to high data inputs the processing is very slow, so we are going to take 1000 x_train and y_train

X_train_, y_train_ = X_train[0:1000], y_train[0:1000]


# In[8]:


svm_clf.fit(X_train_, y_train_) 
svm_clf.predict([some_digit])


# If you call the decision_function() method, you will see that it returns 10 scores per instance (instead of just 1). That’s one score per class:
# 

# In[9]:


some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores


# Here, we can see that the highest score is indeed the one corresponding to class 3

# In[10]:


import numpy as np
np.argmax(some_digit_scores)


# In[11]:


svm_clf.classes_


# In[12]:


svm_clf.classes_[3]


# In[13]:


## Training on SGD classifier
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier()


# In[14]:


sgd_clf.fit(X_train_, y_train_) 

sgd_clf.predict([some_digit])


# In[15]:


sgd_clf.decision_function([some_digit])


# decision function now returns one value per class

# In[16]:


## Evaluating SGDClassifier with the help of cross_val_score()

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train_, y_train_, cv = 3, scoring = "accuracy")


# In[17]:


## Training on Standard Scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_)
cross_val_score(sgd_clf, X_train_scaled, y_train_, cv = 3, scoring = "accuracy")


# # Error Analysis

# In[18]:


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train_, cv=3)


from sklearn.metrics import confusion_matrix
conf_mx = confusion_matrix(y_train_, y_train_pred)
conf_mx


# In[19]:


plt.matshow(conf_mx, cmap = plt.cm.gray)


# In[20]:


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums


# In[21]:


## filling the diagonal with zeros to keep only the errors

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap = plt.cm.gray)
plt.show()

