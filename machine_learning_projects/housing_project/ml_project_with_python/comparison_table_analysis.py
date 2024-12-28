# comparison_table_analysis.py
"""
This file shows how many data-points/rows (districts) pertain to some specific range of median income values.
It also shows the percentual error between the proportions of data-points/rows in these specific ranges 
of the random and stratified test datasets in relation to the full dataset.
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

import data_download
import stratification

# References
data = data_download.data
data_stratified_test = stratification.data_stratified_test
data_train, data_random_test = train_test_split(data, test_size=0.2, random_state=42)

# Comparison Table (Dataset vs. Test Datasets)

# Calculate the district percentages for the full dataset, random test dataset, and stratified test dataset
percentage_data = data["median_income"].value_counts(bins=5, normalize=True).sort_index()
percentage_random = data_random_test["median_income"].value_counts(bins=5, normalize=True).sort_index()
percentage_stratified = data_stratified_test["median_income"].value_counts(bins=5, normalize=True).sort_index()

# Combine the columns into a DataFrame ($ x10,000 vs %)
comparison_table = pd.DataFrame({
    "Dataset Proportion": percentage_data,
    "Random Proportion": percentage_random,
    "Stratified Proportion": percentage_stratified,
})

# Calculate signed percentage errors
comparison_table["Random Error (%)"] = ((comparison_table["Random Proportion"] / comparison_table["Dataset Proportion"]) - 1) * 100
comparison_table["Stratified Error (%)"] = ((comparison_table["Stratified Proportion"] / comparison_table["Dataset Proportion"]) - 1) * 100

# Round the columns to 2 decimal places
comparison_table = comparison_table.round({"Dataset Proportion": 2, "Random Proportion": 2, "Stratified Proportion": 2 })
comparison_table = comparison_table.round({"Random Error (%)": 2, "Stratified Error (%)": 2})

if __name__ == "__main__":
        
    # Dataset: Median Income x Districts
    print("\nMedian Income ($10,000) x % Districts")
    income_districts = data["median_income"].value_counts(bins=5).sort_index() / len(data) * 100
    for income, district in income_districts.items():
        income_range = f"{income.left*10000:.0f} - {income.right*10000:.0f}"
        print(f"$ {income_range}: {district:.2f} %")

    # Dataset x Test Datasets: Median Income

    # Dataset
    print("\nDataset: Median Income (x10,000$) x Districts")
    print(data["median_income"].value_counts(bins=5)) # bins = 5: divides the data in 5 income ranges

    # Random Test Dataset
    print("\nRandomized: Median Income (x10,000$) x Districts")
    print(data_random_test["median_income"].value_counts(bins=5))

    # Stratified Test Dataset
    print("\nStratified: Median Income (x10,000$) x Districts")
    print(data_stratified_test["median_income"].value_counts(bins=5))

    # Dataset x Random x Stratified: percentual error of data-points in the same median income ranges 
    print("\nRnadom vs Stratified (error %):")
    print(comparison_table)
