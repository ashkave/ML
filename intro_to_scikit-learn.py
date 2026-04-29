#!/usr/bin/env python
# coding: utf-8

# ## Getting Data Ready
# ## 1. splitting into features and labels

# In[1]:


#standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# get the dataset
heart_disease = pd.read_csv("heart-disease.csv")
heart_disease.head()


# In[5]:


# from the data set, split the data into features and labels

# features
X = heart_disease.drop("target", axis =1)
X.head()


# In[6]:


# labels
y = heart_disease["target"]
y.head()


# In[10]:


# Splitting into train and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## 2. Convert data into numerical

# In[11]:


# importing car sales dataset which has some non numerical data

car_sales = pd.read_csv("scikit-learn-data/car-sales-extended.csv")
car_sales.head()


# In[12]:


len(car_sales)


# In[13]:


car_sales.dtypes


# In[14]:


# split the data into features and labels

X = car_sales.drop("Price", axis = 1)
y = car_sales["Price"]


# In[15]:


X.head()


# In[16]:


y.head()


# In[20]:


# Split into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[21]:


# turn data from non numerical into numerical
# 1. Import OneHotEncoder and ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 2. Define the categorical features to transform
categorical_features = ["Make", "Colour", "Doors"]

# 3. Create an instance of OneHotEncoder
one_hot = OneHotEncoder()

# 4. Create an instance of ColumnTransformer
transformer = ColumnTransformer([("one_hot", # name
                                  one_hot, # transformer
                                  categorical_features)], # columns to transform
                                  remainder="passthrough") # what to do with the rest of the columns? ("passthrough" = leave unchanged) 

# 5. Turn the categorical features into numbers (this will return an array-like sparse matrix, not a DataFrame)
transformed_X = transformer.fit_transform(X)
transformed_X


# In[23]:


# fit the model with transformed data
from sklearn.ensemble import RandomForestRegressor
np.random.seed(42)

# Create train and test splits with transformed_X
X_train, X_test, y_train, y_test = train_test_split(transformed_X,
                                                    y,
                                                    test_size=0.2)

# Create the model instance
model = RandomForestRegressor()

# Fit the model on the numerical data (this errored before since our data wasn't fully numeric)
model.fit(X_train, y_train)

# Score the model (returns r^2 metric by default, also called coefficient of determination, higher is better)
model.score(X_test, y_test)


# ## 3. Fill Missing Values using Pandas or scikit-learn

# In[27]:


# import dataset with missing values
car_sales_missing = pd.read_csv("scikit-learn-data/car-sales-extended-missing-data.csv")
car_sales_missing.head(10)


# In[28]:


# Get the sum of all missing values
car_sales_missing.isna().sum()


# In[29]:


# Fill the missing values in the Make and colour column
# Note: In previous versions of pandas, inplace=True was possible, however this will be changed in a future version, can use reassignment instead.
# car_sales_missing["Make"].fillna(value="missing", inplace=True)

car_sales_missing["Make"] = car_sales_missing["Make"].fillna(value="missing")
car_sales_missing["Colour"] = car_sales_missing["Colour"].fillna(value="missing")

# Fill the Doors column with the most common value
car_sales_missing["Doors"] = car_sales_missing["Doors"].fillna(value=4)

# Fill the Odometer (KM) column
# Old: car_sales_missing["Odometer (KM)"].fillna(value=car_sales_missing["Odometer (KM)"].mean(), inplace=True)

car_sales_missing["Odometer (KM)"] = car_sales_missing["Odometer (KM)"].fillna(value=car_sales_missing["Odometer (KM)"].mean())


# In[30]:


# Check the number of missing values
car_sales_missing.isna().sum()


# In[31]:


# Remove rows with missing Price labels
car_sales_missing.dropna(inplace=True)


# In[32]:


car_sales_missing.isna().sum()


# In[34]:


# Check the number of total samples (previously was 1000)
len(car_sales_missing)


# In[35]:


# Create features
X_missing = car_sales_missing.drop("Price", axis=1)
print(f"Number of missing X values:\n{X_missing.isna().sum()}")

# Create labels
y_missing = car_sales_missing["Price"]
print(f"Number of missing y values: {y_missing.isna().sum()}")


# In[40]:


# Let's convert the categorical columns to one hot encoded 
# Turn the categories (Make and Colour) into numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"]

one_hot = OneHotEncoder()

transformer = ColumnTransformer([("one_hot", 
                                  one_hot, 
                                  categorical_features)],
                                remainder="passthrough",
                                sparse_threshold=0) # return a sparse matrix or not

transformed_X_missing = transformer.fit_transform(X_missing)
transformed_X_missing


# In[41]:


# Split data into training and test sets
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(transformed_X_missing,
                                                    y_missing,
                                                    test_size=0.2)

# Fit and score a model
model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# ## Filling missing values with scikit-learn

# In[42]:


# import the dataset with missing values

car_sales_missing = pd.read_csv("scikit-learn-data/car-sales-extended-missing-data.csv")
car_sales_missing.head(10)


# In[43]:


#check number of missing values
car_sales_missing.isna().sum()


# In[46]:


#drop missing values from Price which will be used as target
car_sales_missing.dropna(subset=["Price"], inplace = True)


# In[47]:


car_sales_missing.isna().sum()


# In[48]:


# split the data into features and labels

X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]


# In[49]:


X.head()


# In[50]:


y.head()


# In[54]:


# split the data into training and test set

X_train, X_test, y_train, y_test = train_test_split(X,y)
X_train.shape, X_test.shape


# In[56]:


# fill missing values using SimpleImputer
from sklearn.impute import SimpleImputer

# create categorical imputer
cat_imputer = SimpleImputer(strategy = "constant", fill_value = "missing")

# create door imputer
door_imputer = SimpleImputer(strategy = "constant" , fill_value = 4)

# create numeric imputer
num_imputer = SimpleImputer(strategy = "mean")


# In[57]:


# create features
cat_features = ["Make", "Colour"]
door_feature = ["Doors"]
num_features = ["Odometer (KM)"]


# In[59]:


# transform the data

from sklearn.compose import ColumnTransformer

# Create series of column transforms to perform
imputer = ColumnTransformer([
    ("cat_imputer", cat_imputer, cat_features),
    ("door_imputer", door_imputer, door_feature),
    ("num_imputer", num_imputer, num_features)])


# In[60]:


# Find values to fill and transform training data
filled_X_train = imputer.fit_transform(X_train)

# Fill values in to the test set with values learned from the training set
filled_X_test = imputer.transform(X_test)

# Check filled X_train
filled_X_train


# In[61]:


# Get our transformed data array's back into DataFrame's
filled_X_train_df = pd.DataFrame(filled_X_train, 
                                 columns=["Make", "Colour", "Doors", "Odometer (KM)"])

filled_X_test_df = pd.DataFrame(filled_X_test, 
                                columns=["Make", "Colour", "Doors", "Odometer (KM)"])

# Check missing data in training set
filled_X_train_df.isna().sum()


# In[62]:


# Now let's one hot encode the features with the same code as before 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"]

one_hot = OneHotEncoder()

transformer = ColumnTransformer([("one_hot", 
                                  one_hot, 
                                  categorical_features)],
                                remainder="passthrough",
                                sparse_threshold=0) # return a sparse matrix or not

# Fill train and test values separately
transformed_X_train = transformer.fit_transform(filled_X_train_df)
transformed_X_test = transformer.transform(filled_X_test_df)

# Check transformed and filled X_train
transformed_X_train


# In[63]:


# Now we've transformed X, let's see if we can fit a model
np.random.seed(42)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

# Make sure to use the transformed data (filled and one-hot encoded X data)
model.fit(transformed_X_train, y_train)
model.score(transformed_X_test, y_test)


# In[ ]:




