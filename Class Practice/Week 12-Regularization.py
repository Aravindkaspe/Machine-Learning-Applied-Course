#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Read the CSV file-forest fires

# In[3]:


data=pd.read_csv('Data-Week3.csv')
data.head(3)


# Define X and y and split to train and test

# In[6]:


X = data.iloc[:,2:9]
y = data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)


# Define and fit the model

# In[7]:


Lr = LinearRegression()
Lr.fit(X_train,y_train)

#Predict the class of X_test in y
Pred_Lr=Lr.predict(X_test)

Err=np.mean((Pred_Lr-y_test)**2)
Err


# In[ ]:




