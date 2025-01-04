# data_download.py
"""
This module downloads and extracts the dataset.
"""

# Python STL
import os
import tarfile
import urllib.request

# Packages
import pandas as pd

# Data Path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(FILE_DIR)
DATA_PATH = os.path.join(PARENT_DIR, "datasets", "housing")
os.makedirs(DATA_PATH, exist_ok=True)
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"

# Download and extract data
def download_data(data_url=DATA_URL, data_path=DATA_PATH):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    tgz_path = os.path.join(data_path, "housing.tgz")    
    urllib.request.urlretrieve(data_url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(path=data_path)
    data_tgz.close()

# Only download data when this file is run directly
if __name__ == "__main__":
    download_data()

# Load the data
def load_data(data_path=DATA_PATH):
    csv_path = os.path.join(data_path, "housing.csv")
    return pd.read_csv(csv_path)

data = load_data()