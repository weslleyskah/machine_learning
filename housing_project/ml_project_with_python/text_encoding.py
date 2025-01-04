# text_encoding.py
"""
This file shows how to encode a text column into integers and binary vectors for data preprocessing.
"""
import os
import tarfile
import urllib.request
import warnings
warnings.filterwarnings("ignore")
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

import data_download
import stratification

# Dataset
housing = stratification.data_stratified_train.drop(columns=["median_house_value"])
housing_text = housing["ocean_proximity"]

print(housing_text.head(3), "\n")

# Encoding the Text Column: Text -> Int
encoder = LabelEncoder()
housing_text_encoded = encoder.fit_transform(housing_text)

print("Text Encoding (text -> int):\n")
print(housing_text_encoded[0:3])
print(encoder.classes_)
print("'<1H OCEAN' = 0, 'INLAND' = 1, 'ISLAND' = 2, 'NEAR BAY' = 3, 'NEAR OCEAN' = 4\n")

# Binary Encoding: Int -> Binary Vector
encoder = OneHotEncoder()
housing_text_encoded_binary = encoder.fit_transform(housing_text_encoded.reshape(-1,1))

print("Binary Encoding (int -> binary vector):\n")
print(housing_text_encoded_binary.toarray()[0:3])

# Text -> Int -> Binary Vector
encoder = LabelBinarizer()
housing_text_encoded_int_bin = encoder.fit_transform(housing_text)

print(housing_text_encoded_int_bin[0:3])
print(encoder.classes_)
print("'<1H OCEAN' = [1,0,0,0,0], 'INLAND' = [0,1,0,0,0], 'ISLAND' = [0,0,1,0,0], 'NEAR BAY' = [0,0,0,1,0], 'NEAR OCEAN' = [0,0,0,0,1]\n")