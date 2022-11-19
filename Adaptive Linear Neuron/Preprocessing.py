#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
import joblib
warnings.filterwarnings('ignore')


# In[67]:


dataset = pd.read_csv('penguins.csv')
dataset.shape


# In[68]:


dataset.head()


# In[69]:


dataset.isna().sum()


# Since the null values are small and in one feature only, we can make a predictive machine learning model that can classify the gender based on the other features and the label (species) after we process the other features for the best learning, also we can train it only on the species of type Adelie and Gentoo as the null values exist only in them 

# In[70]:


dataset['species'].unique()


# In[71]:


mask = dataset['gender'].isnull()==True
nullEntries = dataset[mask]


# In[72]:


nullEntries


# Checking data normality in python: https://www.youtube.com/watch?v=12qn03Ml87E
# 
# Perform data transformation for normality: https://www.youtube.com/watch?v=xOZ3DxybQKc 
# 
# All the features are skewed and have a negative kurtosis but feature (bill_length) is the closest to normal distribution

# In[73]:


numericalFeatures=['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']


# In[74]:


def getAndVisualizeSkewness(feature):
    print("Visualizing feature {feature}".format(feature=feature))
    plt.hist(dataset[feature])
    plt.show()
    print("{feature} is skewed by {skew}".format(feature = feature, skew = stats.skew(dataset[feature])))
    print("{feature}'s kurtosis is {kurtosis}".format(feature=feature,kurtosis=stats.kurtosis(dataset['bill_length_mm'])))
    print('-'*40)


# In[75]:


for feature in numericalFeatures:
    getAndVisualizeSkewness(feature)


# All of these transformation functions will also result in a skewed data, so another solution is to transform according to different measures,
# 
# body mass is in grams -> convert to kilograms
# 
# flipper length in millimeters -> convert to meters
# 
# scaling would not affect skewness, but at least all the numbers will be in the same range

# In[76]:


def transform(feature):
    print("{0}".format(feature))
    transformedFeature = dataset[feature].transform([np.sqrt,np.log,np.reciprocal])
    transformedFeature.hist(layout=(2,2),figsize=(8,8))
    plt.show()
    print('-'*40)


# In[77]:


for feature in numericalFeatures:
    transform(feature)


# In[78]:


dataset['body_mass_g'] = dataset['body_mass_g']/1000
dataset['flipper_length_mm'] = dataset['flipper_length_mm']/1000
dataset['bill_length_mm'] = dataset['bill_length_mm']/100
dataset['bill_depth_mm'] = dataset['bill_depth_mm']/100


# In[79]:


dataset


# In[80]:


speciesWithNullEntriesMask = dataset['species'] != 'Chinstrap'
speciesWithNullEntries = dataset[speciesWithNullEntriesMask]
speciesWithNullEntries.head()


# In[81]:


nullEntries = speciesWithNullEntries[mask]
nullEntries


# In[82]:


nullEntriesIndex = nullEntries.index
nullEntriesIndex


# In[83]:


speciesWithNullEntries['species'] = [1 if species =='Adelie' else 0 for species in speciesWithNullEntries['species']]
nullEntries['species'] =  [1 if species =='Adelie' else 0 for species in nullEntries['species']]
speciesWithNullEntries


# In[84]:


speciesWithNullEntries.drop(index=nullEntriesIndex, axis=0,inplace=True)


# In[85]:


speciesWithNullEntries.isna().sum()


# In[86]:


speciesWithNullEntries['gender'] = [1 if gender=='female' else 0 for gender in speciesWithNullEntries['gender']]


# In[87]:


genderModel = LogisticRegression()
y_train = speciesWithNullEntries['gender']
x_train = speciesWithNullEntries.drop(columns=['gender'],axis=1)
x_test = nullEntries.drop(columns=['gender'],axis=1)
x_test


# In[88]:


genderModel.fit(x_train,y_train)
y_train_predicted = genderModel.predict(x_train)
print("the accuracy is {0}".format(accuracy_score(y_train,y_train_predicted)))
y_test = genderModel.predict(x_test)
y_test


# In[89]:


joblib.dump(genderModel,"genderModel.h5")


# In[90]:


nullEntries['gender'] = y_test


# In[91]:


speciesWithNullEntries = speciesWithNullEntries.append([nullEntries])
speciesWithNullEntries


# In[92]:


speciesWithNullEntries['species'] = ['Adelie' if species ==1 else 'Gentoo' for species in speciesWithNullEntries['species']]
speciesWithNullEntries ['gender']= ['female' if gender==1 else 'male' for gender in speciesWithNullEntries['gender']]


# In[93]:


speciesWithNullEntries


# In[94]:


chinstrap = dataset[dataset['species']=='Chinstrap']
processedDataset = pd.DataFrame()
processedDataset = processedDataset.append([speciesWithNullEntries,chinstrap])
processedDataset['gender'] = [1 if gender =='female' else 0 for gender in processedDataset['gender']]
processedDataset


# In[95]:


processedDataset.isna().sum()


# In[96]:


processedDataset.to_csv("processed-penguins.csv",index = False)


# In[ ]:




