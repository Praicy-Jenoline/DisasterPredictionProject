#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils/preprocessing.py
import pandas as pd

def clean_features(df: pd.DataFrame):
    """Preprocess incoming features (normalize, handle missing)"""
    df = df.fillna(0)
    return df
