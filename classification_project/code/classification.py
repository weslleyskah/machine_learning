# Python STL
import os
import tarfile
import urllib.request
import shutil
import math

# Packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# SKLearn: Datasets, ML Algorithms 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score





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




# Cross-validation
def cross_val_score(dataset, dataset_label):

    """
    Perform stratified cross-validation on a given dataset and compute accuracy for each fold.

    Parameters:
    dataset : pd.DataFrame or np.ndarray
        The input dataset containing the features. Can be a pandas DataFrame or a NumPy array. 
        If it is a DataFrame, it will be converted to a NumPy array.
    
    dataset_label : pd.DataFrame or np.ndarray
        The labels corresponding to the dataset. Can be a pandas DataFrame or a NumPy array.
        If it is a DataFrame, it will be converted to a NumPy array and flattened.

    Returns:
    scores : list of float
        A list of accuracy scores, one for each fold of the stratified cross-validation.

    Description:
    This function performs a 3-fold stratified cross-validation to evaluate the performance of
    a Stochastic Gradient Descent (SGD) classifier on the provided dataset. The process involves:
    1. Splitting the dataset into 3 stratified folds.
    2. For each fold:
        - Training the classifier on the training set.
        - Predicting the labels for the test set.
        - Calculating the accuracy (proportion of correct predictions) on the test set.
    3. Appending the accuracy score for each fold to a list.
    4. Returning the list of accuracy scores.
    """
    
    if (isinstance(dataset, pd.DataFrame) and isinstance(dataset_label, pd.DataFrame)):
        dataset = dataset.to_numpy()
        dataset_label = dataset_label.to_numpy().ravel()

    # Ensure inputs are NumPy arrays
    dataset = np.asarray(dataset)
    dataset_label = np.asarray(dataset_label).ravel()

    # Create a Stochastic Gradient Descent classifier
    binary_classifier = SGDClassifier(random_state=42)
    # Create a StratifiedKFold object with 3 subsets (folds)
    skfolds = StratifiedKFold(n_splits=3, shuffle=True)
    
    # List to store accuracy for each fold
    scores = [] 
    
    """
    For each fold, the dataset is split into training and testing subsets. The classifier is trained on the training subset
    and evaluated on the testing subset. The accuracy of the classifier for each fold is calculated and stored in the scores list.

    Parameters:
    dataset (numpy.ndarray): The entire dataset to be used for cross-validation.
    dataset_label (numpy.ndarray): The labels corresponding to the dataset.
    binary_classifier (sklearn.base.BaseEstimator): The classifier to be evaluated.
    skfolds (sklearn.model_selection.StratifiedKFold): The K-fold cross-validation splitter.
    scores (list): A list to store the accuracy scores for each fold.

    Process:
    1. Split the dataset into training and testing subsets for each subset (fold).
    2. Clone the classifier to ensure each fold uses a fresh instance.
    3. Train the classifier on the training subset.
    4. Predict the labels for the testing subset.
    5. Calculate the accuracy of the predictions.
    6. Append the accuracy score to the scores list.
    """
    for train_index, test_index in skfolds.split(dataset, dataset_label):
        dataset_train_fold = dataset[train_index]
        dataset_train_label_fold = dataset_label[train_index]

        dataset_test_folds = dataset[test_index]
        dataset_label_test_folds = dataset_label[test_index]

        clone_clf = clone(binary_classifier)
        clone_clf.fit(dataset_train_fold, dataset_train_label_fold)
        predictions = clone_clf.predict(dataset_test_folds)
        accuracy = np.mean(predictions == dataset_label_test_folds)
        scores.append(accuracy)

    return scores, sum(scores) / len(scores)



# Confusion matrix
def k_fold_confusion_matrix(dataset_train, dataset_label_train, classifier, prediction):
    """
    Perform K-fold cross-validation and compute the confusion matrix, precision, recall, and F1 score.

    Parameters:
    dataset_train (numpy.ndarray): The training dataset.
    dataset_label_train (numpy.ndarray): The labels for the training dataset.
    classifier (sklearn.base.BaseEstimator): The classifier to be used for predictions.
    prediction (numpy.ndarray): The predictions made by the classifier.

    Returns:
    tuple: A tuple containing the confusion matrix, precision score, recall score, and F1 score.

    Precision (float): The ratio of true positive predictions to the total number of positive predictions made.
                       It indicates the accuracy of the positive predictions.
    Recall (float): The ratio of true positive predictions to the total number of actual positives.
                    It indicates the ability of the classifier to find all positive samples.
    F1 Score (float): The harmonic mean of precision and recall, providing a single metric that balances both concerns.
                      It is useful when you need a balance between precision and recall.
    """
    skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    dataset_predictions = cross_val_predict(classifier, dataset_train, dataset_label_train, cv=skfolds)
    
    conf_matrix = confusion_matrix(dataset_label_train, dataset_predictions)
    precision = precision_score(dataset_label_train, dataset_predictions)
    recall = recall_score(dataset_label_train, dataset_predictions)
    harmonic_mean_precision_recall = f1_score(dataset_label_train, dataset_predictions)
    
    return conf_matrix, precision, recall, harmonic_mean_precision_recall



if __name__ == "__main__":

    # Download data
    dataset_download()

    # Load data
    dataset, dataset_label = dataset_load()





    # Split into training and test subsets
    dataset_train, dataset_label_train = dataset[:60000], dataset_label[:60000]
    dataset_test, dataset_label_test = dataset[60000:], dataset_label[60000:]





    # Shuffle / Randomize the dataset
    # random_indices = [ ? ?? ??? ... ? = [0-len(dataset)-1] xlen(dataset) or [0-59999] x60000 ] 
    # dataset_train = [ dataset_train[?] dataset_train[??] ... ]
    random_indices = np.random.permutation(len(dataset_train))
    dataset_train = dataset_train.iloc[random_indices].reset_index(drop=True)
    dataset_label_train = dataset_label_train.iloc[random_indices].reset_index(drop=True)





    # Random digit
    digit_random = dataset_train.iloc[1].to_numpy()
    digit_random_image = digit_random.reshape(28, 28)
    plt.imshow(digit_random_image, cmap=plt.cm.binary, interpolation="nearest")
    plt.title(f"Label: {dataset_label_train.iloc[1]}")
    plt.savefig(os.path.join(CLASSIFICATION_IMAGES_DIR, "random_digit.png"))
    print(f"Rnadom digit figure save on {CLASSIFICATION_IMAGES_DIR}.")





    # Binary Classification

    # Binary Classifier: digit image 5 = True, other digit images = False
    dataset_label_train_5 = (dataset_label_train == 5)
    dataset_label_test_5 = (dataset_label_test == 5)
    
    # Stochastic Gradient Descent Classifier
    """
        Algorithm.fit(dataset, dataset_label)
        Dataframes: dataset.to_numpy() dataset_label.to_numpy().ravel()
        Arrays: dataset_label.ravel()
    """
    binary_classifier = SGDClassifier(random_state=42)
    binary_classifier.fit(dataset_train.to_numpy(), dataset_label_train_5.to_numpy().ravel())
    # Prediction
    prediction = binary_classifier.predict([digit_random])
    print("Prediction (True if 5, else False):", prediction[0], dataset_label_train.iloc[1])
    # Evaluation
    accuracy = binary_classifier.score(dataset_test.to_numpy(), dataset_label_test_5)
    print(f"Test Accuracy: {accuracy}")





    # Performance Measure using Cross Validation
    print("""The cross_val_score function performs cross-validation by splitting the dataset into subsets (folds) to evaluate the performance of the classifier.
    Scores of the cross validation:""")
    scores, average_scores = cross_val_score(dataset_train, dataset_label_train_5)
    for score in scores: 
        print(score, end=", ")
    print(average_scores)





    # K-fold Valuation and Confusion Matrix
    conf_matrix, precision, recall, harmonic_mean_precision_recall = k_fold_confusion_matrix(
        dataset_train.to_numpy(), dataset_label_train_5.to_numpy().ravel(), binary_classifier, prediction
    )
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nPrecision Score (percentage of correct predictions):")
    print(precision)
    print("\nRecall Score (percentage of the detection of the positive class):")
    print(recall)
    print("\nHarmonic mean of the prediction and recall scores:")
    print(harmonic_mean_precision_recall)

