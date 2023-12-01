#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("housing.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.dropna(inplace=True)


# In[6]:


data.isnull().sum()


# In[7]:


df=pd.DataFrame(data)
df


# In[8]:


x=df.drop(["median_house_value"],axis=1)
x


# In[9]:


y=df["median_house_value"]
y


# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=1)


# In[11]:


train_df=x_train.join(y_train)
train_df


# In[12]:


train_df.hist(figsize=(50,25))


# In[13]:


plt.figure(figsize=(15,8))
sns.heatmap(train_df.corr(),annot=True, cmap="YlGnBu")


# In[14]:


train_df["total_rooms"]=np.log(train_df["total_rooms"]+1)
train_df["total_bedrooms"]=np.log(train_df["total_bedrooms"]+1)
train_df["population"]=np.log(train_df["population"]+1)
train_df["households"]=np.log(train_df["households"]+1)


# In[15]:


train_df.hist(figsize=(50,25))


# In[16]:


train_df.ocean_proximity.value_counts()


# In[17]:


train_df=train_df.join(pd.get_dummies(train_df.ocean_proximity)).drop(["ocean_proximity"],axis=1)


# In[18]:


train_df


# In[19]:


plt.figure(figsize=(15,8))
sns.heatmap(train_df.corr(),annot=True, cmap="YlGnBu")


# In[20]:


plt.figure(figsize=(15,8))
sns.scatterplot(x="latitude",y="longitude",data=train_df, hue="median_house_value",palette="coolwarm")


# In[21]:


train_df["bedroom_ratio"]=train_df["total_bedrooms"]/train_df["total_rooms"]
train_df["household_rooms"]=train_df["total_rooms"]/train_df["households"]


# In[22]:


plt.figure(figsize=(15,8))
sns.heatmap(train_df.corr(),annot=True, cmap="YlGnBu")


# In[23]:


from sklearn.linear_model import LinearRegression

x_train, y_train= train_df.drop(["median_house_value"],axis=1),train_df["median_house_value"]

reg=LinearRegression()

reg.fit(x_train, y_train)


# In[24]:


test_df=x_test.join(y_test)

test_df["total_rooms"]=np.log(test_df["total_rooms"]+1)
test_df["total_bedrooms"]=np.log(test_df["total_bedrooms"]+1)
test_df["population"]=np.log(test_df["population"]+1)
test_df["households"]=np.log(test_df["households"]+1)

test_df=test_df.join(pd.get_dummies(test_df.ocean_proximity)).drop(["ocean_proximity"],axis=1)

test_df["bedroom_ratio"]=test_df["total_bedrooms"]/test_df["total_rooms"]
test_df["household_rooms"]=test_df["total_rooms"]/test_df["households"]



test_df


# In[25]:


x_test, y_test= test_df.drop(["median_house_value"],axis=1),test_df["median_house_value"]


# In[26]:


reg.score(x_test,y_test)*100


# In[27]:


from sklearn.svm import SVR


# In[28]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

svm_reg=SVR(kernel='linear')

svm_reg.fit(x_train, y_train)

# Testing the model

svm_reg.score(x_test,y_test)*100


# In[29]:


from sklearn.ensemble import RandomForestRegressor


# In[30]:


forest= RandomForestRegressor()

forest.fit(x_train, y_train)


# In[32]:


forest.score(x_test, y_test)*100


# In[ ]:




