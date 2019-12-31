#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv(r"D:\DOWNLOADS\New folder\housing.csv")


# ## Taking a quick look at the data structure
# 

# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing.ocean_proximity.value_counts()


# In[6]:


housing.describe()


# In[7]:


## Plotting histogram for each numberical attribute
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins = 60, figsize = (20,10))


# ## Creating a test set

# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# ## Discovering and Visualizing the data to Gain insights

# In[10]:


## Visualizing Geographical data
housing.plot(kind = 'scatter', x ='longitude', y = 'latitude')


# In[11]:


## alpha = 0.1 is used to visualize the places where there is a high density of data points
housing.plot(kind = 'scatter', x ='longitude', y = 'latitude', alpha = 0.1)


# ## Looking for Correlations

# In[12]:


corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending = False)


# Here, we can see the correlation betweeen median house values(label) and other attributes.

# In[13]:


## Checking the correlation between attributes by using scatter_matrix() function
from pandas.plotting import scatter_matrix
attributes = ['housing_median_age', 'total_rooms','median_income',
       'median_house_value']
scatter_matrix(housing[attributes], figsize = (15,7))


# Here, we have no plotting scatter_plot for all attributes because 11 attributes will require 121 plots, which will not be fitting in the page

# Moreover, here, we can see that for this problem median_income is most promising attribute to predict median house value

# In[14]:


## So, we are now plotting scatter plot to chech the correlation between median income vs median house value
housing.plot(kind ='scatter', x = "median_income", y ="median_house_value", alpha = .1)


# This plot reveals a few things. First, the correlation is indeed very strong; you can clearly see the upward
# trend and the points are not too dispersed. Second, the price cap that we noticed earlier is clearly visible
# as a horizontal line at $500,000. 
# 
# But this plot reveals other less obvious straight lines: a horizontal line
# around $450,000, another around $350,000, perhaps one around $280,000, and a few more below that.
# You may want to try removing the corresponding districts to prevent your algorithms from learning to
# reproduce these data quirks.

# ## Experimenting with the Attribute Combinations
# The dataset shows that total number of rooms in district is not very useful attribute when we don't know total number of households there are. So, we require number of rooms per household. 
# 
# Similarly, total number of bedrooms is also not useful. So, we need to compare to to the number of rooms.
# 
# Population per household is also an interesting attribute. So, lets create new attributes:
# 
# 
# 

# In[15]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]


# In[16]:


corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending = False)


# The new bedrooms_per_room attribute is much more correlated with the median house
# value than the total number of rooms or bedrooms. Apparently houses with a lower bedroom/room ratio
# tend to be more expensive. The number of rooms per household is also more informative than the total
# number of rooms in a district — obviously the larger the houses, the more expensive they are.

# # Preparing the data for ML algorithms

# ## 1. Data Cleaning

# You noticed earlier that the total_bedrooms attribute has some missing values, so
# let’s fix this. You have three options:
# 
# 
# Get rid of the corresponding districts.
# 
# Get rid of the whole attribute.
# 
# Set the values to some value (zero, the mean, the median, etc.).

# In[17]:


## splitting labels and features
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()


# In[18]:


## Scikit-Learn provides a handy class to take care of missing values: Imputer.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")


# Since the median can only be computed on numerical attributes, we need to create a copy of the data
# without the text attribute ocean_proximity:

# In[19]:


housing_num = housing.drop("ocean_proximity", axis = 1)


# In[20]:


## Now you can fit the imputer instance to the training data using the fit() method:
imputer.fit(housing_num)


# In[21]:


print("Imputer_stats:", imputer.statistics_)


# In[22]:


housing_num.median().values


# In[23]:


## Now you can use this “trained” imputer to transform the training set by replacing missing values by the learned medians:
X = imputer.transform(housing_num)


# In[24]:


## The result is a plain Numpy array containing the transformed features. If you want to put it back into aPandas DataFrame, it’s simple:
housing_tr = pd.DataFrame(X, columns= housing_num.columns, index = housing_num.index)


# In[25]:


housing_tr.head()
housing_tr.shape


# ## 2. Handling Text and Categorical Attributes
# 
# Earlier we left out the categorical attribute ocean_proximity because it is a text attribute so we cannot
# compute its median. Most Machine Learning algorithms prefer to work with numbers anyway, so let’s
# convert these text labels to numbers.

# In[26]:


## Looking at the first 10 values of ocean_proximity
housing_cat = housing["ocean_proximity"]
housing_cat.head()


# In order to work with this dataset, we can convert these categories from text to numbers with the help of Sklearn's LabelEncoder and  OneHotEncoder class:

# In[27]:


## Using Label encoder to transform text into number

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat_enco= encoder.fit_transform(housing_cat)
housing_cat_enco


# In[28]:


## converting text in ocean_proximity feature to numbers
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_hot = encoder.fit_transform(housing_cat_enco.reshape(-1,1))
housing_cat_hot


# You can use it mostly like a normal 2D array,but if you really want to convert it to a (dense) NumPy array, just call the toarray() method:

# In[29]:


housing_cat_hot.toarray()


# In[30]:


### We can apply both transformations (from text categories to integer categories, then from integer categories
### to one-hot vectors) in one shot using the LabelBinarizer class:

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot


# ## 3. Custom Transformers

# In[31]:


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

#Class Constructor
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
#Return self, nothing else to do here
    def fit(self, X, y=None):
        return self # nothing else to do
    
#Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]

#Check if needed     
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# ## 4. Transformation Pipelines

# In[32]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[33]:


## Pipeline for the numerical attributes:
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),  ## Imputer
    ('attribs_adder', CombinedAttributesAdder()),  
    ('std_scaler', StandardScaler() ),    ## Transformer
])

## Fitting and transforming the pipeline
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[34]:


## However, the new dversion Scikit Learn has been able to provide pipline for both numerical and categorical data in one go

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs),
])


# In[35]:


housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape


# # Select and Train a Model

# ## 1. Training and Evaluating on the Training set

# In[36]:


## Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[37]:


### Instances from Linear Regession

some_data = housing.iloc[:5]
some_labels = housing_labels[:5]

some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))


# In[38]:


print("Labels:", list(some_labels))


# From here, we can see actual values and predicted values of housing_price_values, which is not accurate. Now, we will check error

# In[39]:


## RMSE

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_predictions, housing_labels)

lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[40]:


## Decision Tree Regressor (training the model)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)



# In[41]:


## Evaluating on training set
from sklearn.metrics import mean_squared_error


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# ## 2. Better Evaluation Using Cross-Validation

# 
# 
# 

# ## K-Fold Cross-validation feature
# It randomly splits the training set into 10 (cv = 10) distinct subsets called folds, then it trains and
# evaluates the Decision Tree model 10 times, picking a different fold for evaluation every time and
# training on the other 9 folds.

# In[42]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)
tree_rmse_scores = np.sqrt(-scores)


# In[43]:


## Displaying the scores

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Std_Deviation:", scores.std())


# In[44]:


display_scores(tree_rmse_scores)


# Now the Decision Tree doesn’t look as good as it did earlier. In fact, it seems to perform worse than the
# Linear Regression model!

# In[47]:


## Let’s compute the same scores for the Linear Regression model just to be sure:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)
lin_rmse_scores = np.sqrt(-lin_scores)


## Displaying the scores
display_scores(tree_rmse_scores)


# The Decision Tree model is overfitting so badly that it performs worse than the Linear
# Regression model. So, now lets try wil Random Forest Model

# ## Random Forest Model 
# Random Forests work by training many Decision Trees on random subsets of the features, then averaging out their
# predictions. Building a model on top of many other models is called Ensemble Learning, and it is often a
# great way to push ML algorithms even further.
# 

# In[48]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)


# In[51]:


forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[52]:


## Let’s compute the same scores for the Random Forest model just to be sure:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)
forest_rmse_scores = np.sqrt(-forest_scores)


## Displaying the scores
display_scores(forest_rmse_scores)


# Random Forests look very promising. However, note that the score on the
# training set is still much lower than on the validation sets, meaning that the model is still overfitting the
# training set. Possible solutions for overfitting are to simplify the model, constrain it (i.e., regularize it), or
# get a lot more training data.

# # Fine Tuning
# 
# Assuming, we have shortlisted promising models. Now, we are required to fine-tune them:
# 

# ## 1. Grid Search

# In[61]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [30, 40, 60], 'max_features':[3,5,7,9]},
    {'bootstrap': [False], 'n_estimators': [30, 40], 'max_features': [3,5,7]},
    ]


# In[62]:


forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = 'neg_mean_squared_error', return_train_score= True)

grid_search.fit(housing_prepared, housing_labels)


# In[63]:


## Best combinations of the parameters

grid_search.best_params_


# In[64]:


## finding the best estimator by:

grid_search.best_estimator_


# In[65]:


## Evaluation Scores
cvres = grid_search.cv_results_


for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# Here, we have found the best solution by setting max_features hyperparameter to 9 and n_estimators hyperparameter to 60.
# 
# 
# RMSE score for this combination is 49596.3250105323

# # Analyzing the best models and their errors

# In[66]:


feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances


# In[67]:


## Displaying these important scores next to their corresponding attribute names

extra_attribs = ["rooms_per_hhold", 'pop_per_hhold', 'bedrooms_per_room']
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted (zip(feature_importances, attributes), reverse = True)


# # Evaluate your system on the test set

# In[69]:


final_model = grid_search.best_estimator_

X_test = test_set.drop("median_house_value", axis = 1)
y_test = test_set["median_house_value"]


# In[70]:


X_test_prepared = full_pipeline.transform(X_test)


# In[75]:


final_predictions.shape


# In[82]:


y_test = test_set["median_house_value"]
y_test.shape


# In[83]:


final_predictions = final_model.predict(X_test_prepared)


final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[ ]:




