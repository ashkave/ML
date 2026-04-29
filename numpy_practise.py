#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# ## Datatypes and Attributes

# In[3]:


# NumPy's main datatype is an ndarray
array1d = np.array([1,2,3])


# In[4]:


array1d


# In[5]:


array2d = np.array([[1,2,3],
               [4,5,6]])


# In[6]:


array2d


# In[7]:


array3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])


# In[8]:


array3d


# In[9]:


array1d.shape


# In[10]:


array2d.shape


# In[11]:


array3d.shape


# In[12]:


array3d.ndim


# In[13]:


type(array3d)


# In[14]:


array3d.size


# In[15]:


array3d.dtype


# In[16]:


import pandas as pd
df = pd.DataFrame(array2d)
df


# In[17]:


ones = np.ones((3,3))


# In[18]:


ones


# In[19]:


zeroes = np.zeros((2,2))


# In[20]:


zeroes


# In[21]:


random_array = np.random.randint(0,9,size=(3,3))


# In[22]:


random_array


# In[23]:


#pseudo random number. how to keep the random numbers from changing when sharing the JN to others
np.random.seed(seed=0)
random_array2 = np.random.randint(0,9,size=(5,3))
random_array2


# ## Viewing arrays and matrices

# In[24]:


np.unique(random_array)


# ## Manipulating Arrays

# ### Arithmetic Operations

# In[25]:


a1 = np.array([7,8,9])
a1


# In[26]:


a2 = np.array([[4,5,6],[1,2,3],[3,3,9]])
a2


# In[27]:


a3 = np.array([[[1,2,3],[2,3,4]],[[2,3,4],[3,4,5]],[[4,5,6],[5,6,7]]])
a3


# In[28]:


a3.shape


# In[29]:


a2.shape


# In[30]:


a1 * a2


# In[31]:


ones = np.ones([3,4])


# In[32]:


ones


# In[33]:


ones = np.ones(3)


# In[34]:


ones


# In[35]:


a1


# In[36]:


a1 + ones


# In[37]:


a1 - ones


# In[38]:


a1 * a2


# In[39]:


a1


# In[40]:


a2


# In[41]:


a2*a3


# In[42]:


a2 = np.array([[1,2,3],[2,3,4]])


# a2 

# In[43]:


a2


# In[44]:


a2 * a3


# In[45]:


a2 / a3


# In[46]:


a2 //a3


# In[47]:


a2 % 2


# ## Aggregation

# ### if working with python datatypes use python functions but if working with numpy arrays use numpy functions as they are much faster

# In[48]:


large_array = np.random.random(100000)
large_array.size


# In[49]:


large_array[:10]


# In[50]:


get_ipython().run_line_magic('timeit', "sum(large_array) #python's sum")
get_ipython().run_line_magic('timeit', "np.sum(large_array) # np's sum")


# In[51]:


a2


# In[52]:


np.mean(a2)


# In[53]:


np.max(a2)


# In[54]:


np.min(a2)


# ### Mean is the sum of values divided by number of values

# In[55]:


a1 = np.array([4,5.6, 2,3, 6.8])


# In[56]:


a1


# In[57]:


np.mean(a1)


# ### SD is how far the numbers are spread out from the mean. It is the squareroot of variance

# In[58]:


np.std(a1)


# ### Variance is avg degree to which each number is different from the mean

# In[59]:


np.var(a1)


# In[60]:


np.sqrt(np.var(a1))


# In[61]:


low_var_array = np.array([2,4,6,8,10])


# In[62]:


low_var_array


# In[63]:


high_var_array = np.array([1,2,3,1000,2000,3000,200,300,400])


# In[64]:


high_var_array


# In[65]:


np.var(low_var_array)


# In[66]:


np.var(high_var_array)


# In[67]:


np.mean(low_var_array)


# In[68]:


np.mean(high_var_array)


# In[69]:


np.std(low_var_array)


# In[70]:


np.std(high_var_array)


# In[71]:


import matplotlib.pyplot as plt


# In[72]:


plt.hist(high_var_array)


# In[73]:


plt.hist(low_var_array)


# In[ ]:




