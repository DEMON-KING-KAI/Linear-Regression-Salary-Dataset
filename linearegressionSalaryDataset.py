#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("C:/Users/DITU/Desktop/Salary_Data.csv")


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


plt.figure(figsize = (15,10))
plt.scatter(df['YearsExperience'], df["Salary"])


# In[6]:


X = df[['YearsExperience']]
y = df[['Salary']]


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 1)


# In[8]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()


# In[9]:


lr.fit(X_train, y_train)


# In[10]:


y_pred = lr.predict(X_test)


# In[11]:


lr.coef_


# In[12]:


lr.intercept_


# In[13]:


plt.figure(figsize = (15,10))
plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_test, y_pred, color = 'green')
plt.plot(X_train, lr.predict(X_train), color = 'black')
plt.title('Salary vs Experience (Result)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()


# In[14]:


import statsmodels.api as sm

X_stat = sm.add_constant(X_train)
regsummary = sm.OLS(y_train, X_stat).fit()
regsummary.summary()


# In[15]:


from sklearn.metrics import r2_score

r2_score(y_train, lr.predict(X_train))


# In[ ]:




