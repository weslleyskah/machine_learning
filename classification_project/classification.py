# Python STL
import os
import tarfile
import urllib.request
import shutil
import math

# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset
from sklearn.datasets import fetch_openml

# Load the dataset
dataset_load = fetch_openml("mnist_784", version=1)
print(dataset_load)

if __name__ == "__main__":
    # Extract data and labels
    dataset, dataset_label = dataset_load["data"], dataset_load["target"]
    print(dataset)
    print("\n")
    print(dataset_label)
    
    # Access a specific digit by row index
    digit_random = dataset.iloc[36000].to_numpy()  # .iloc to select the row in the Dataframe and then convert to NumPy array
    digit_random_image = digit_random.reshape(28, 28)  # reshape to 28x28 pixels
    plt.imshow(digit_random_image, cmap=plt.cm.binary, interpolation="nearest")
    plt.title(f"Label: {dataset_label[36000]}")  
    plt.show()