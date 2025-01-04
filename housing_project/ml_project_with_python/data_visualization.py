# data_visualization.py
"""
This file shows how to create a graph with multiple columns of the dataset as visual elements of the graph.
"""
import os
import tarfile
import urllib.request
import shutil
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

data = data_download.data
housing = stratification.data_stratified_train

# Data Visualization
def housing_visualization():
    housing.plot(kind="scatter", x = "longitude", y = "latitude", alpha=0.4,
                 s=housing["population"]/100, label="population",                 # s = size of the data-points
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True) # c = color of the data-points
    plt.legend()
    plt.show()

if __name__ == "__main__":
    housing_visualization()