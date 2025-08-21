#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils/preprocessing.py
from sklearn.preprocessing import StandardScaler

def scale_features(dataframe):
    """Scales the dataset features using Standard Scaler."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dataframe)
    return scaled

