# data_display.py
"""
This module modifies the printing format on the terminal of the dataset.
"""

# Python STL
import os
import tarfile
import urllib.request
import shutil

# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modules
import data_download

# Show all columns and adjust terminal width
pd.set_option("display.max_columns", None)                           
pd.set_option("display.width", shutil.get_terminal_size().columns)

# Example: run file
if __name__ == "__main__":
    data = data_download.data
    print(data.iloc[0:1])