# data_cleaning.py
"""
This file shows how to fill the missing column values of the dataset with the median of each respective column.
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
from sklearn.impute import SimpleImputer

import data_download
import stratification

# Dataset
housing = stratification.data_stratified_train.drop(columns=["median_house_value"])
# Column Label
housing_labels = stratification.data_stratified_train["median_house_value"].copy() 
# Numerical dataset
housing_numerical = housing.drop(columns=["ocean_proximity"])

# Impute the median of each column into all the the missing column values
def impute_median(dataset_numerical):
    imputer = SimpleImputer(strategy="median")
    imputer.fit(dataset_numerical)
    dataset_transformed = imputer.transform(dataset_numerical)
    dataset_df = pd.DataFrame(dataset_transformed, columns = housing_numerical.columns)
    return dataset_df

if __name__ == "__main__":

    housing_transformed = impute_median(housing_numerical)

    # Missing Values
    print(housing_numerical.isnull().sum())
    print(pd.isnull(housing_transformed).sum())
    