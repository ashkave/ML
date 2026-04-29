#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


car_series = pd.Series(["BMW","Toyota","Honda","Skoda"])


# In[3]:


car_series


# Series is a 1d data type

# In[4]:


colour_series = pd.Series(["Red","Yellow","Blue","Pink"])


# In[5]:


colour_series


# In[6]:


df = pd.DataFrame({"car-make" : car_series, "colour":colour_series})


# In[7]:


df


# In[9]:


car_data = pd.read_csv("car-sales.csv")


# In[10]:


car_data


# ## Describe Data

# In[11]:


car_data.dtypes


# In[12]:


car_data.columns


# In[13]:


car_columns = car_data.columns


# In[14]:


car_columns


# In[15]:


car_data.describe()


# In[16]:


car_data.info()


# In[17]:


car_data.index


# ## Viewing and Selecting Data
# 

# In[18]:


car_data.head()


# In[19]:


car_data.head(7)


# In[20]:


car_data.tail()


# In[ ]:




