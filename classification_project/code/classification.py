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

# Datasets
from sklearn.datasets import fetch_openml

# Ml Algorithms
from sklearn.linear_model import SGDClassifier




# Paths
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_FILE_DIR = os.path.dirname(FILE_DIR)
PARENT_DIR = os.path.dirname(PARENT_FILE_DIR)
## Datasets
DATA_PATH = os.path.join(PARENT_DIR, "datasets")
CLASSIFICATION_DATA_PATH = os.path.join(DATA_PATH, "classification_data")
## Models
MODEL_DIR = os.path.join(PARENT_DIR, "models")
CLASSIFICATION_MODEL_DIR = os.path.join(MODEL_DIR, "classification_models")
## Images
IMAGES_DIR = os.path.join(PARENT_DIR, "img")
CLASSIFICATION_IMAGES_DIR = os.path.join(IMAGES_DIR, "classification_img")

directories = [DATA_PATH, CLASSIFICATION_DATA_PATH, MODEL_DIR, CLASSIFICATION_MODEL_DIR, IMAGES_DIR, CLASSIFICATION_IMAGES_DIR]
for dir in directories:
    os.makedirs(dir, exist_ok=True)




# Download Data
def dataset_download():

    dataset_load = fetch_openml("mnist_784", version=1)

    dataset_data = dataset_load["data"]
    dataset_label = dataset_load["target"]

    dataset_data.to_csv(os.path.join(CLASSIFICATION_DATA_PATH, "mnist_data.csv"), index=False)
    dataset_label.to_csv(os.path.join(CLASSIFICATION_DATA_PATH, "mnist_label.csv"), index=False)

    print(f"Dataset downloaded and saved as CSV files inside {CLASSIFICATION_DATA_PATH}.")

def dataset_load():

    dataset_data_csv = os.path.join(CLASSIFICATION_DATA_PATH, "mnist_data.csv")
    dataset_label_csv = os.path.join(CLASSIFICATION_DATA_PATH, "mnist_label.csv")

    dataset_data_df = pd.read_csv(dataset_data_csv)
    dataset_label_df = pd.read_csv(dataset_label_csv)

    return dataset_data_df, dataset_label_df 




if __name__ == "__main__":
    # Download data
    # dataset_download()

    # Load data
    dataset, dataset_label = dataset_load()

    # Split into training and test subsets
    dataset_train, dataset_label_train = dataset[:60000], dataset_label[:60000]
    dataset_test, dataset_label_test = dataset[60000:], dataset_label[60000:]

    # Shuffle / Randomize the dataset
    shuffle_index = np.random.permutation(len(dataset_train))
    dataset_train = dataset_train.iloc[shuffle_index].reset_index(drop=True)
    dataset_label_train = dataset_label_train.iloc[shuffle_index].reset_index(drop=True)

    # Binary Classifier: Digit 5 = True, other digits = False
    dataset_label_train_5 = (dataset_label_train == 5)
    dataset_label_test_5 = (dataset_label_test == 5)

    # Stochastic Gradient Descent Classifier
    sgd_clf = SGDClassifier(random_state=42)
    # sgd_clf.fit(dataset_train, dataset_label_train_5)
    sgd_clf.fit(dataset_train.to_numpy(), dataset_label_train_5.values.ravel())

    # Test the model with a random digit
    digit_random = dataset_train.iloc[36000].to_numpy()
    prediction = sgd_clf.predict([digit_random])
    print("Prediction (True if 5, else False):", prediction[0])

    # Evaluate accuracy
    accuracy = sgd_clf.score(dataset_test.to_numpy(), dataset_label_test_5)
    print(f"Test Accuracy: {accuracy}")

    # Show a random image from the training set
    digit_random_image = digit_random.reshape(28, 28)
    plt.imshow(digit_random_image, cmap=plt.cm.binary, interpolation="nearest")
    plt.title(f"Label: {dataset_label_train.iloc[36000]}")
    plt.show()
