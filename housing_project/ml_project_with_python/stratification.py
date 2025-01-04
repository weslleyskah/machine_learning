# stratification.py
"""
This module constructs a smaller stratified test dataset from the dataset.
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
from sklearn.model_selection import StratifiedShuffleSplit

# Modules
import data_download

# References
data = data_download.data

# Stratification

def stratify_dataset(dataset):
    # 1: Add Stratification Column: Stratification column based on a dataset column
    # It divides the median_income column values in 5 ranges according to 5 values on the income-category column
    # np.ceil: gives the integer greater than or equal to the result of the division
    # dataset["median_income"] / 1.5: you divide by 1.5 to scale the median_income values to a smaller range
    # .where(condition, value_changed): you limit the maximum value of the income-category column to 5 
    # if the condition is True, the value will remain the same, but if False, the value will be changed
    dataset["income-category"] = np.ceil(dataset["median_income"] / 1.5)
    dataset["income-category"].where(dataset["income-category"] < 5, 5.0)

    # 2: Stratification Algorithm
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(dataset, dataset["income-category"]):
        dataset_stratified_train = dataset.loc[train_index]
        dataset_stratified_test = dataset.loc[test_index]

    # 3: Clean up: Remove stratification column from all the datasets
    for set_ in (dataset, dataset_stratified_train, dataset_stratified_test):
        set_.drop("income-category", axis=1, inplace=True)
    
    return dataset_stratified_train, dataset_stratified_test

data_stratified_train, data_stratified_test = stratify_dataset(data)

if __name__ == "__main__":
    print("Dataset (rows, columns):", data.shape)
    print("Stratified train dataset:", data_stratified_train.shape)
    print("Stratified test dataset:", data_stratified_test.shape)