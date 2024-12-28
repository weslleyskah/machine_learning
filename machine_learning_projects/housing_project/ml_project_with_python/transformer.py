# transformer.py
"""
This module contains the classes/transformers holding functions that modify the dataset  
for the full data transformation pipeline to preprocess numerical and categorical/text data.
"""

# Python STL
import os
import tarfile
import urllib.request
import math

# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Test Datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Tranformation Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

# ML Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Modules
import data_download
import data_display
import stratification

# Selector: to select the numerical and text columns from a dataset
class CustomDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, dataset, dataset_label = None):
        return self
    def transform(self, dataset, dataset_label = None):
        return dataset[self.columns].values

# Text Encoder: to convert text columns into binary vectors
class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, categories = None):
        self.categories = categories
        self.label_binarizer = None

    def fit(self, dataset, dataset_label = None):
        if self.categories is not None:
            self.label_binarizer = LabelBinarizer()
            self.label_binarizer.fit(self.categories) # text encode the specific text categories
        else:
            self.label_binarizer = LabelBinarizer()
            self.label_binarizer.fit(dataset)        # text encode the text column values that appear on the dataset
        return self                                  # warning: smaller subsets may not show all the possible text categories from the dataset
    
    def transform(self, dataset, dataset_label = None):
        return self.label_binarizer.transform(dataset)

# Custom Transformer Class: group of functions to modify the dataset
class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def fit(self, dataset, dataset_label = None):
        return self
    
    # Creation of new columns
    def transform(self, dataset, dataset_label = None):

        rooms_index, population_index, households_index = 3, 5, 6
        rooms_per_household = dataset[:, rooms_index] / dataset[:, households_index] 
        population_per_household = dataset[:, population_index] / dataset[:, households_index]
        
        return np.c_[dataset, rooms_per_household, population_per_household]