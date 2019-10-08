#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv("https://gist.githubusercontent.com/omarish/5687264/raw/7e5c814ce6ef33e25d5259c1fe79463c190800d9/mpg.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.columns


# In[7]:


df.shape


# In[11]:


df.dtypes


# In[12]:


df.horsepower.unique()


# In[15]:


print(df[pd.to_numeric(df["horsepower"], errors='coerce')])


# In[16]:


print(df[pd.to_numeric(df["horsepower"], errors='coerce').isnull()])


# In[21]:


print(df[pd.to_numeric(df["mpg"], errors='coerce').isnull()])


# In[23]:


df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')


# In[25]:


df.dtypes


# In[29]:


df.horsepower.value_counts().sum()


# In[30]:


df.mpg.value_counts().sum()


# In[31]:


df.displacement.value_counts().sum()


# In[33]:


df.cylinders.value_counts().sum()


# In[34]:


df.isnull().values.any()


# In[35]:


cols = df.columns


# In[36]:


df[cols] = df[cols].apply(pd.to_numeric, errors = 'coerce')


# In[37]:


df.isna().values.any()


# In[42]:


df.applymap(np.isreal)


# In[43]:


df.head()


# In[44]:


df.describe()


# In[45]:


df.count()


# In[50]:


print(df.drop(columns =['name', 'origin', 'model_year']))


# In[51]:


df


# In[52]:


df.replace('?', 'Nan')


# In[71]:


df.dropna(inplace = True)


# In[97]:


x =  pd.DataFrame(np.c_[df["cylinders"], df["horsepower"], df["weight"]], columns =["cylinders", "horsepower", "weight"] )
print(x)


# In[88]:


y = df["mpg"].values.reshape(-1,1)
print(y)


# In[98]:


df.dropna()


# In[99]:


x.shape


# In[100]:


y.shape


# In[101]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)


# In[102]:


reg = LinearRegression()


# In[103]:


reg.fit(x_train, y_train)


# In[104]:


print("interceptor value:", reg.intercept_)
print("slope:", reg.coef_)


# In[105]:


y_pred =reg.predict(x_test)


# In[110]:


##comparing actual output with predicted output

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1 = df.head(50)
df1.plot(kind = 'bar', figsize = (20,10))
plt.show()


# In[121]:


from sklearn import metrics
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  ### Note that for rmse, the lower that value is, the better the fit

print("R^2 Score:", r2_score(y_test, y_pred)) # The closer towards 1, the better the fit


# In[133]:


plt.scatter(x_test["horsepower"], y_test)

plt.show()


# In[134]:


plt.scatter(x_test["weight"], y_test)
plt.show()


# In[114]:


plt.scatter(x_test["cylinders"], y_test)
plt.show()


# In[124]:


reg = LinearRegression()
reg.fit(x_train[['horsepower']], y_train)


# In[128]:


reg = LinearRegression()
reg.fit(x_train[['weight']], y_train)


# In[129]:


reg = LinearRegression()
reg.fit(x_train[['cylinders']], y_train)


# In[126]:


y_predicted = reg.predict(x_test[['horsepower']])
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('R²: %.2f' % r2_score(y_test, y_predicted))


# In[130]:


y_predicted = reg.predict(x_test[['weight']])
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('R²: %.2f' % r2_score(y_test, y_predicted))


# In[131]:


y_predicted = reg.predict(x_test[['cylinders']])
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('R²: %.2f' % r2_score(y_test, y_predicted))


# In[ ]:





# In[127]:


reg = LinearRegression()
reg.fit(x_train[['horsepower','weight','cylinders']], y_train)
y_predicted = reg.predict(x_test[['horsepower','weight','cylinders']])
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('R²: %.2f' % r2_score(y_test, y_predicted))

