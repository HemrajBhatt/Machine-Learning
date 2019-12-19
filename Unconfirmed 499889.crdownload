#!/usr/bin/env python
# coding: utf-8

# # Predicting whether a passenger on the titanic would have been survived or not.

# ## 1. Importing the Important libraries

# In[1]:


## Importing linear alzebra, data processing, data visualization 
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## Importing algorithms

from sklearn.naive_bayes import GaussianNB


# ## 2. Uploading the data

# In[3]:


test_df = pd.read_csv(r"C:\Users\hemra\Downloads\New folder\test.csv")
train_df = pd.read_csv(r"C:\Users\hemra\Downloads\New folder\train.csv")


# ## 3. Data Exploration

# In[4]:


train_df.info()


# The training-set has 891 examples and 11 features + the target variable (survived).  2 of the features are floats, 5 are integers and 5 are objects. 

# In[5]:


train_df.describe()


# Above we can see that 38% out of the training-set survived the Titanic. We can also see that the passenger ages range from 0.4 to 80. On top of that we can already detect some features, that contain missing values, like the ‘Age’ feature.

# In[6]:


train_df.head(10)


# From the table above, we can note a few things. First of all, that we need to convert a lot of features into numeric ones later on, so that the machine learning algorithms can process them. Furthermore, we can see that the features have widely different ranges, that we will need to convert into roughly the same scale. We can also spot some more features, that contain missing values (NaN = not a number), that wee need to deal with.

# In[7]:


## Identifying the actual missing data
titanic = train_df.isnull().sum().sort_values(ascending=False)
print(titanic)


# In[8]:


percent_1 = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending=False)
print(percent_1)


# In[9]:


percent_2 = (round(percent_1, 1))
print(percent_2)


# In[10]:


missing_data = pd.concat([titanic, percent_2], axis=1, keys=['Missing_data', '%'])
missing_data.head(5)


# From the above table, we can see that Embarked feature has only 2 missing values, which can easily be filled. It will be much more tricky, to deal with the ‘Age’ feature, which has 177 missing values. The ‘Cabin’ feature needs further investigation, but it looks like that we might want to drop it from the dataset, since 77 % of it are missing.

# In[11]:


## Identifying the names of columns
train_df.columns


# Above you can see the 11 features + the target variable (survived). What features could contribute to a high survival rate ?

# To me it would make sense if everything except ‘PassengerId’, ‘Ticket’ and ‘Name’ would be correlated with a high survival rate

# 1. Survived and Not-survived
# 

# In[12]:


##Creating a plot between the passengers who survived and not survived
sns.countplot(x = "Survived", data = train_df)


# In[13]:


##to find male and female survived and not survived
sns.countplot(x = "Survived", hue = 'Sex', data = train_df)


# hue = "Sex" means that the out of survived and not survived how many are male and female
# 

# In[14]:


## hue as  passenger class 
sns.countplot(x = "Survived", hue= "Pclass", data = train_df)


# In[15]:


## 1. Age and Sex:
survived = 'survived'
not_survived = 'not survived'


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15, 5))

women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')


ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
ax.set_title('Male')


# From the above graph, we can see that men have a high probability of survival when they are between 18 and 30 years old, which is also a little bit true for women but not fully. For women the survival chances are higher between 14 and 40.
# For men the probability of survival is very low between the age of 5 and 18, but that isn’t true for women. Another thing to note is that infants also have a little bit higher probability of survival.
# Since there seem to be certain ages, which have increased odds of survival and because I want every feature to be roughly on the same scale, I will create age groups later on.
# 
# 

# In[16]:


##2. Embarked, Pclass and Sex:
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4, aspect=1.2)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()


# In[17]:


## 3. Pclass 
sns.barplot(x='Pclass', y='Survived', data=train_df)


# In[18]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# The plot above confirms our assumption about pclass 1, but we can also spot a high probability that a person in pclass 3 will not survive.
# 

# 4. SibSp and Parch:
# SibSp and Parch would make more sense as a combined feature, that shows the total number of relatives, a person has on the Titanic. I will create it below and also a feature that sows if someone is not alone.
#   
# 

# In[19]:


data = [train_df, test_df]
for dataset in data:
    dataset["relatives"] = dataset['SibSp'] + dataset["Parch"]
    dataset.loc[dataset['relatives']>0, 'not_alone'] = 0
    dataset.loc[dataset['relatives']==0, 'not_alone'] = 1
    dataset['not_alone']= dataset['not_alone'].astype(int)
    
train_df['not_alone'].value_counts()


# In[20]:


axes = sns.factorplot('relatives','Survived', 
                      data=train_df, aspect = 2.5, )


# In tha above graph, we can see that you had a high probabilty of survival with 1 to 3 realitves, but a lower one if you had less than 1 or more than 3 (except for some cases with 6 relatives).
# 

# ## 4. Data Preprocessing
# 

# In[21]:


## First, I will drop ‘PassengerId’ from the train set, because it does not contribute to a persons survival probability. I will not drop it from the test set, since it is required there for the submission.
train_df = train_df.drop(['PassengerId'], axis=1)


# Missing Data:
# 
# 1. Cabin:
# As a reminder, we have to deal with Cabin (687), Embarked (2) and Age (177). First I thought, we have to delete the ‘Cabin’ variable but then I found something interesting. A cabin number looks like ‘C123’ and the letter refers to the deck. Therefore we’re going to extract these and create a new feature, that contains a persons deck. Afterwords we will convert the feature into a numeric variable. The missing values will be converted to zero. In the picture below you can see the actual decks of the titanic, ranging from A to G.

# In[22]:


train_df.head(10)


# In[23]:


import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)


# In[24]:


# we can now drop the cabin feature
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# 2. Age:
# Now we can tackle the issue with the age features missing values. I will create an array that contains random numbers, which are computed based on the mean age value in regards to the standard deviation and is_null.

# In[25]:


data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    
# compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    
# fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
    
train_df["Age"].isnull().sum()


# 3. Embarked:
#     
# Since the Embarked feature has only 2 missing values, we will just fill these with the most common one.

# In[26]:


train_df['Embarked'].describe()


# In[27]:


common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)


# ## Converting Features:
# 

# In[28]:


train_df.info()


# Above you can see that ‘Fare’ is a float and we have to deal with 4 categorical features: Name, Sex, Ticket and Embarked. Lets investigate and transfrom one after another.
# 

# 1. Fare:
# Converting “Fare” from float to int64, using the “astype()” function pandas provides:

# In[29]:


data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# 2. Sex:
# Convert ‘Sex’ feature into numeric.

# In[30]:


genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# 3. Ticket: 

# In[31]:


train_df['Ticket'].describe()


# Since the Ticket attribute has 681 unique tickets, it will be a bit tricky to convert them into useful categories. So we will drop it from the dataset.
# 

# In[32]:


train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)


# 4. Embarked:
# Convert ‘Embarked’ feature into numeric.

# In[33]:


ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# ## Creating Categories:
# 

# 1. Age:
# Now we need to convert the ‘age’ feature. First we will convert it from float into integer. Then we will create the new ‘AgeGroup” variable, by categorizing every age into a group. Note that it is important to place attention on how you form these groups, since you don’t want for example that 80% of your data falls into group 1

# In[34]:


data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


# In[35]:


# let's see how it's distributed train_df['Age'].value_counts()
train_df['Age'].value_counts()


# 2. Fare:
# For the ‘Fare’ feature, we need to do the same as with the ‘Age’ feature. But it isn’t that easy, because if we cut the range of the fare values into a few equally big categories, 80% of the values would fall into the first category. Fortunately, we can use sklearn “qcut()” function, that we can use to see, how we can form the categories.

# In[36]:


train_df.head(10)


# In[37]:


data = [train_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[38]:


train_df['Fare'].value_counts()


# ## Creating new Features
# I will add two new features to the dataset, that I compute out of other features.
# 

# In[39]:


## 1. Age times Class
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']


# In[40]:


## 2. Fare per Person
data = [train_df, test_df]

for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)


# In[41]:


train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)


# In[42]:


# Let's take a last look at the training set, before we start training the models.
train_df.head(10)


# In[43]:


test_df.head()


# # 5. Building Machine Learning Models
# 

# Now we will train several Machine Learning models and compare their results. Note that because the dataset does not provide labels for their testing-set, we need to use the predictions on the training set to compare the algorithms with each other. Later on, we will use cross validation.
# 

# In[44]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()


# ## Gaussian Naive Bayes:
# 

# In[45]:


gaussian = GaussianNB() 


# In[46]:


gaussian.fit(X_train, Y_train)


# In[47]:


Y_pred = gaussian.predict(X_test)


# In[48]:


acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)


# Let us try to imporve its accuracy by different algorithms. For this, we have to import differnt algorithms:
# 

# In[49]:


# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC


# ## Logistic Regression

# In[50]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)


# In[51]:


Y_pred = logreg.predict(X_test)


# In[52]:


acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)


# ## K Nearest Neighbor
# 

# In[53]:


knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train)  
Y_pred = knn.predict(X_test)


# In[54]:


acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)


# ## Perceptron
# 

# In[55]:


perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)


# In[56]:


acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)


# ## Linear Support Vector Machine
# 

# In[57]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)


# In[58]:


acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)


# ## Decision Tree
# 

# In[59]:


decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)


# In[60]:


acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)


# ## Random Forest

# In[61]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)


# In[62]:


random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)


# ## Stochastic Gradient Descent (SGD)
# 

# In[63]:


sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)


# In[64]:


sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd)


# ## The best Model

# In[65]:


results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')

result_df.head(9)


# As we can see, the Random Forest classifier goes on the first place. But first, let us check, how random-forest performs, when we use cross validation.
# 

# ## K-Fold Cross Validation
# K-Fold Cross Validation randomly splits the training data into K subsets called folds. Let’s image we would split our data into 4 folds (K = 4). Our random forest model would be trained and evaluated 4 times, using a different fold for evaluation everytime, while it would be trained on the remaining 3 folds.
# 
# The code below perform K-Fold Cross Validation on our random forest model, using 10 folds (K = 10). Therefore it outputs an array with 10 different scores.
# 
# 
# 

# In[68]:


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")


# In[69]:


print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# This looks much more realistic than before. Our model has a average accuracy of 81% with a standard deviation of 4 %. The standard deviation shows us, how precise the estimates are .
# This means in our case that the accuracy of our model can differ + — 4%.
# 
# 
