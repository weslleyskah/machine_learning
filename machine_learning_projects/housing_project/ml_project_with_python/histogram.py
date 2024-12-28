# histogram.py
"""
This file shows how to create a graph of the data-points / number of rows (Y) and the ranges of values from a column (X).
It shows how many rows have values from a specific column inside some specific range.
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

data = data_download.data

# Histogram of the Dataset
data.hist(bins=20, figsize=(10, 5))
plt.tight_layout()  
plt.show()

# Histogram for the median house value
data_scaled = data["median_house_value"] / 10000
_, bin_edges, _ = plt.hist(data_scaled, bins=20, edgecolor="grey", color="orange")
plt.grid(True, axis='both', linestyle='-', color='grey', alpha=0.5)
plt.xlabel("Median House Value (x $10,000)")  
plt.ylabel("Number of Districts")  
plt.title("Histogram of Median House Value")  

# Range labels
plt.xticks(bin_edges, rotation=45)  

plt.show()

# Histogram for the median income
data["median_income"].hist(bins=20, figsize=(10, 6), edgecolor="black")
plt.xlabel("Median Income (x $10,000)")  
plt.ylabel("Number of Districts")  
plt.title("Histogram of Median Income") 

# Range labels
bin_edges = plt.hist(data["median_income"], bins=20)[1] 
plt.xticks(bin_edges, rotation=45)  

plt.show()