#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


dataset = pd.read_csv(r"C:\Users\Dell\Downloads\Weather.csv")


# In[7]:


dataset.shape


# In[8]:


dataset.describe()


# In[10]:


dataset.isna()


# In[12]:


dataset.columns


# In[15]:


dataset.replace(' ', "nan")


# In[17]:


dataset.drop(columns = ['SNF', 'SND', 'FT', 'FB', 'FTI', 'ITH', 'PGT', 'TSHDSBRSGF', 'SD3', 'RHX', 'RHN', 'RVG', 'WTE'], inplace = True)


# In[21]:


dataset.columns


# In[29]:


dataset.drop(columns = ['SPD'], inplace = True)


# In[30]:


dataset


# In[31]:


##predicting the maximum temperature taking input feature as minimum temperature



# In[42]:


x = dataset['MinTemp'].values.reshape(-1,1)
print(x)
x.dtype


# In[ ]:


MaxTemp	MinTemp	


# In[41]:


y = dataset['MaxTemp'].values.reshape(-1,1)
print(y, y.dtype)


# In[43]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# In[44]:


regressor = LinearRegression()


# In[45]:


regressor.fit(x_train, y_train)


# In[46]:


print(regressor.intercept_)


# In[47]:


print(regressor.coef_)


# In[48]:


## Till above steps, we have trained our algorithm, Now its time to make some predictions


# In[50]:


y_pred =  regressor.predict(x_test)
print(y_pred)


# In[51]:


# comparing the predicted values with actual o/p

df=pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)


# In[55]:


#plotting a bar graph to compare actual and predicted values
df1 =df.head(30)
df1.plot(kind = 'bar', figsize = (20,10))
plt.grid(which='major', linstyle='-', linewidth = '0.5', color = '%g')
plt.grid(which='minor', linstyle=':', linewidth = '0.5', color = '%b')
plt.show()


# In[56]:


# now plot a  straight line with test data

plt.scatter(x_test, y_test, color = "gray")
plt.plot(x_test, y_pred, color = "red", linewidth = 2)
plt.show()


# In[60]:


# last step is to evaluate the performance of the alorithm
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


## value of root mean squared error is 4.19, 
### which is more than 10% of the mean value of the percentages of all the temperature i.e. 23.888.
#### This means that our algorithm was not very accurate but can still make reasonably good predictions.


# In[ ]:





# In[ ]:




