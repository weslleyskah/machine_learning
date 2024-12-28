# linear_correlation.py
"""
This file shows how to create a graph with the linear function between 2 columns.
"""
import os
import tarfile
import urllib.request
import hashlib
import warnings
warnings.filterwarnings("ignore")
import math
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import data_download

# References
data = data_download.data
housing = data.drop(columns=["ocean_proximity"])

# Linear Correlation
# Columns vs Median House Value
corr_matrix = housing.corr()
linear_corr = corr_matrix["median_house_value"].sort_values(ascending=False)
print(linear_corr)

columns = housing.copy()

for col in columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(housing[col], housing["median_house_value"], alpha=0.5, c="gray")
    plt.title(f"{col} vs. median_house_value")
    plt.xlabel(col)
    plt.ylabel("median_house_value")
    plt.show()
# Median Income vs Median House Value
columns_2 = ["median_house_value", "median_income"]
scatter_matrix(housing[columns_2], figsize=(10,6))
plt.show()