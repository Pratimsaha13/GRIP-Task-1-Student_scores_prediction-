#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("C:/Users/Partha/Desktop/student_scores - student_scores.csv")
df.head()


# In[3]:


plt.bar(df["Hours"],df["Scores"],data=df)


# In[4]:


plt.scatter(x=df["Hours"],y=df["Scores"])


# In[5]:


X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[7]:


X_train, X_test ,y_train,y_test = train_test_split(X ,y , test_size=0.2 , random_state=0)


# In[8]:


clf = LinearRegression()
clf.fit(X_train ,y_train)


# In[9]:


y_pred = clf.predict(X_test)
print(y_pred)


# In[10]:


# Plotting the regression line
line = clf.coef_*X+clf.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[11]:


df1 = pd.DataFrame({"Actual":y_test,"Predicted_Scores":y_pred})
df1


# In[12]:


# You can also test with your own data
hours = np.array(9.25)
pred = clf.predict(hours.reshape(-1,1))
print("No of Hours = "+str(hours))
print("Predicted Score = "+str(pred[0]))


# In[13]:


from sklearn import metrics
MAE = metrics.mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error: "+str(MAE))


# In[ ]:




