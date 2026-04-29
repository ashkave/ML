#!/usr/bin/env python
# coding: utf-8

# In[85]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[86]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[87]:


plt.plot()


# In[88]:


plt.plot();


# In[89]:


x =[1,2,3,4,5]
y = [10,20,30,40,50]
plt.plot(x,y);


# In[90]:


fig, ax = plt.subplots();
ax.plot(x,y);


# ## SAMPLE WORKFLOW

# In[91]:


# 1. import the libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[92]:


# 2. Prepare the data

x = [ 1,2,3,4,5,6]
y = [100,200,300,400,500,700]


# In[93]:


# 3. create figure
fig, ax = plt.subplots()


# In[94]:


# 4. fill data
ax.plot(x,y);


# In[95]:


ax.set(title = "Simple Plot", xlabel="items",ylabel = "prices")


# In[96]:


fig.savefig("images/simple-plot.png")


# ## numpy arrays visualization

# In[97]:


import numpy as np


# In[98]:


x = np.linspace(0,10,100)


# In[99]:


x[:10]


# In[100]:


x


# In[101]:


# line plot
fig, ax = plt.subplots()
ax.plot(x, x*2)
plt.show();


# In[102]:


# scatter plot
fig, ax = plt.subplots()
ax.scatter(x, np.exp(x))
plt.show();


# In[103]:


fig, ax = plt.subplots()
ax.scatter(x, np.sin(x))
plt.show();


# In[104]:


item_prices = { "laptop" : "$650", "desktop" : "$900", "tablet" : "$300"}


# In[105]:


item_prices


# In[106]:


# vertical bar plot
fig,ax = plt.subplots()
ax.bar(item_prices.keys(), item_prices.values());
ax.set(title = "ITEM PRICES");


# In[ ]:





# In[107]:


# horizontal bar plot
fig,ax = plt.subplots()
ax.barh(list(item_prices.keys()), list(item_prices.values()));


# In[108]:


#histogram
fg , ax = plt.subplots()
ax.hist(np.random.randn(1000));


# In[109]:


#subplots
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols =2, figsize=(10,5));
ax1.plot([1,2,3,4],[11,22,33,44]);
ax2.scatter(np.random.random(10), np.random.random(10));
ax3.barh(list(item_prices.keys()), list(item_prices.values()));
ax4.hist(np.random.randn(1000));


# ## Plotting data directory with pandas

# In[112]:


import pandas as pd
car_sales = pd.read_csv("car-sales.csv")
car_sales


# In[115]:


car_sales["Price"] = car_sales["Price"].str.replace(r'[\$\,]', '', regex=True).astype(float)
car_sales


# In[116]:


car_sales["Price"]


# In[118]:


# Add a date column
car_sales["Sale Date"] = pd.date_range("1/1/2026", periods=len(car_sales))
car_sales


# In[119]:


car_sales["Total Sales"] = car_sales["Price"].cumsum()
car_sales


# In[120]:


car_sales.plot(x='Sale Date', y='Total Sales');


# In[121]:


car_sales.plot(x="Odometer (KM)", y="Price", kind='scatter');


# In[122]:


x = np.random.rand(10,4)
x


# In[123]:


df = pd.DataFrame(x, columns = ["a","b","c","d"])
df.plot.bar();


# In[124]:


car_sales.plot(x='Make', y='Odometer (KM)', kind='bar');


# In[125]:


car_sales["Odometer (KM)"].plot.hist();


# In[127]:


car_sales["Odometer (KM)"].plot(kind="hist");


# In[128]:


car_sales["Price"].plot.hist(bins=20);


# In[ ]:




