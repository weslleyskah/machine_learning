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
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
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




def cross_val_score(dataset, dataset_label):
    
    dataset = dataset.to_numpy()
    dataset_label = dataset_label.to_numpy()

    sgd_clf = SGDClassifier(random_state=42)
    skfolds = StratifiedKFold(n_splits=3, shuffle=True)
    
    for train_index, test_index in skfolds.split(dataset, dataset_label):

        dataset_folds = dataset[train_index]
        dataset_label_folds = dataset_label[train_index]

        dataset_test_folds = dataset[test_index]
        dataset_label_test_folds = dataset_label[test_index]

        clone_clf = clone(sgd_clf)
        clone_clf.fit(dataset_folds, dataset_label_folds.ravel())
        predictions = clone_clf.predict(dataset_test_folds)
        predictions_correct = sum(predictions == dataset_label_test_folds)

        print(predictions_correct / len(predictions))





if __name__ == "__main__":

    """
        Algorithm.fit(dataset, dataset_label)
        Dataframes: dataset.to_numpy() dataset_label.to_numpy().ravel()
        Arrays: dataset_label.ravel()
    """

    # Download data
    # dataset_download()

    # Load data
    dataset, dataset_label = dataset_load()

    # Split into training and test subsets
    dataset_train, dataset_label_train = dataset[:60000], dataset_label[:60000]
    dataset_test, dataset_label_test = dataset[60000:], dataset_label[60000:]

    # Shuffle / Randomize the dataset
    # random_indices = [ ? ?? ??? ... ? = [0-59999] x60000 ] 
    # dataset_train = [ dataset_train[?] dataset_train[??] ... ]
    random_indices = np.random.permutation(len(dataset_train))
    dataset_train = dataset_train.iloc[random_indices].reset_index(drop=True)
    dataset_label_train = dataset_label_train.iloc[random_indices].reset_index(drop=True)




    # Binary Classification: Test a ML Model on an image

    # Binary Classifier: Digit 5 = True, other digits = False
    dataset_label_train_5 = (dataset_label_train == 5)
    dataset_label_test_5 = (dataset_label_test == 5)

    # Test the model with a random digit
    digit_random = dataset_train.iloc[36000].to_numpy()
    
    # Stochastic Gradient Descent Classifier
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(dataset_train.to_numpy(), dataset_label_train_5.to_numpy().ravel())
    # Prediction
    prediction = sgd_clf.predict([digit_random])
    print("Prediction (True if 5, else False):", prediction[0])
    # Evaluation
    accuracy = sgd_clf.score(dataset_test.to_numpy(), dataset_label_test_5)
    print(f"Test Accuracy: {accuracy}")

    # Show the random digit
    digit_random_image = digit_random.reshape(28, 28)
    plt.imshow(digit_random_image, cmap=plt.cm.binary, interpolation="nearest")
    plt.title(f"Label: {dataset_label_train.iloc[36000]}")
    plt.show()




    # Performance Measure using Cross Validation
    cross_val_score(dataset_train, dataset_label_train_5)
