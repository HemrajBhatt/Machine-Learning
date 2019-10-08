#!/usr/bin/env python
# coding: utf-8

# # Step 1: Collecting the data
# 

# In[1]:



## Importing differnt libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import sklearn


# In[2]:


titanic_data = pd.read_csv(r"C:\Users\Dell\Downloads\2dfd2de0d4f8727f873422c5d959fff5-ff414a1bcfcba32481e4d4e8db578e55872a2ca1\Statistics_ML_26Aug-master/titanic_data.csv")


# In[3]:


titanic_data.head(10)


# In[4]:


# finding the number of passengers in the ship

titanic_data.PassengerId.count()


# # Step 2: Analyzing the Data
# 
# 
# 

# In[5]:


##Creating a plot between the passengers who survived and not survived
sns.countplot(x = "Survived", data = titanic_data)


# In[6]:


## to find male and female survived and not survived
sns.countplot(x = "Survived", hue = 'Sex', data = titanic_data)
## hue = "Sex" means that the out of survived and not survived how many are male and female


# In[7]:


## hue as  passenger class 
sns.countplot(x = "Survived", hue= "Pclass", data = titanic_data)


# In[8]:


## Now we are going to do age distribution
titanic_data.Age.plot.hist()


# In[9]:


## ploting histogram of fare size
titanic_data.Fare.plot.hist(bins = 20, figsize = (10,5))


# In[10]:


## now finding what columns are still left
titanic_data.info()


# In[11]:


sns.countplot(x = "SibSp", data = titanic_data)


# In[12]:


sns.countplot(x = "Parch", data = titanic_data)


# # Step 3: Data Wranggling 

# In[13]:


# There might me some null values which requires special treatment
## first checking whether the data is null
titanic_data.isnull()


# In[14]:


# checking which column has how many null values
titanic_data.isnull().sum()


# In[15]:


## to visualize the quantity of missing data through heatmap
sns.heatmap(titanic_data.isnull())


# In[16]:


## now considering the age column
sns.boxplot(x="Pclass", y = "Age", data = titanic_data)
#Now on the basis of below box plot we can see that passengers in class 3 are younger than other two class. 
### Therefore, now we will be perfom imputation (which is either replacing the null value with mean or removing the row)


# In[17]:


titanic_data.head()


# In[18]:


titanic_data.drop("Cabin", axis = 1, inplace = True)


# In[19]:


titanic_data.head()


# In[20]:


## Now deleting NA values
titanic_data.dropna(inplace = True) 


# In[21]:


sns.heatmap(titanic_data.isnull())


# In[22]:


## see wherther still there is null value

titanic_data.isnull().sum()


# In[23]:


titanic_data.head()


# In[24]:


## we can see that there is huge string or continous data but we require categorical data for logistic regression so we will convert
### WE will convert to some dummy variables. This is done with the help of  pandas


# In[25]:


## getting dummy variables
pd.get_dummies(titanic_data.Sex)


# In[26]:


## now we don't require both the columns because it is understandable through a single column
## So we will drop first column

sex =pd.get_dummies(titanic_data.Sex, drop_first = True)

sex.head()


# In[27]:


##Similar we can also do for the columns that are haing string values
embark =pd.get_dummies(titanic_data.Embarked, drop_first = True)## here we have doped first column becaus other two are enough for telling about the data
embark.head()


# In[28]:


Pcl =pd.get_dummies(titanic_data.Pclass, drop_first = True)
Pcl.head()


# In[29]:


## Now we have made the values categorical
## So, Now we are requried to concatenate all these new values
new_data = pd.concat([sex,Pcl,embark], axis = 1)
new_data.head()


# In[30]:


## So, Now we are requried to concatenate all these new values into a dataset
titanic = pd.concat([titanic_data, new_data], axis = 1)
titanic.head()


# In[31]:


titanic.drop(columns = ['Sex', 'Name', 'Ticket', 'Embarked','PassengerId'], inplace = True)


# In[32]:


titanic.head()


# In[33]:


titanic.drop(columns = ['Pclass'], inplace = True)


# In[34]:


# Here Data wranglling is completed and we have acquired the data required for training and testing
titanic.head()


# # Step 3 and 4: Training and Testing the data
# ## Here we are going to split the data set into train and test subsets and then build the model on train data. Then, predict the output on test data set

# ## 1. Train Data : defining the dependent and independent data

# In[41]:


X =titanic.drop("Survived", axis = 1)
y = titanic.Survived 


# #### Spliting the Data into training and testing subset

# In[42]:


import sklearn


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[46]:


from sklearn.linear_model import LogisticRegression


# In[48]:


logmodel = LogisticRegression()


# In[49]:


logmodel.fit(X_train, y_train)


# ## 2: Test Data: Predicting the model on test data (how accurate is our model)
# 

# In[51]:


predictions = logmodel.predict(X_test)


# ### how our model has been performing we can calculate accuracy or classification report

# In[55]:


from sklearn.metrics import classification_report
classification_report(y_test, predictions)


# ### We can also use confusion matrix accuracy
# 

# In[59]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions) 
### confusion matrix
##     PN  PY
## AN  105 21
## AY  25  63


# In[60]:


## now to caculate the accuracy: accuracy = (105+53)/(105+21+25+63), but we dont have to do manually rather:
from sklearn.metrics import accuracy_score


# In[61]:


accuracy_score(y_test, predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




