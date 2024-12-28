# column_division.py
"""
This file shows how to divide columns to create new columns / attributes for the dataset.
"""
import os
import tarfile
import urllib.request
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
import data_display

# References
data = data_download.data
housing = data.drop(columns=["ocean_proximity"])

# Column Division
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]


if __name__ == "__main__":

    # Dataset with new columns
    print(housing.iloc[0:1])

    # Linear Correlation
    corr_matrix = housing.corr()
    linear_corr_median_house_value = corr_matrix["median_house_value"].sort_values(ascending=False)
    print(linear_corr_median_house_value)

    # Scatter Plot
    cols = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
    for col in cols:
        plt.scatter(housing[col], housing["median_house_value"])
        plt.show()