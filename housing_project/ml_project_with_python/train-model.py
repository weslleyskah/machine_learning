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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Tranformation Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

# ML Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Save Model
import joblib

# Fine tune Model
from sklearn.model_selection import GridSearchCV

# Modules
import data_download
import data_display
import stratification
import transformer
import transformation_pipeline

# References
full_pipeline = transformation_pipeline.full_pipeline
data_stratified_train = stratification.data_stratified_train
data_stratified_test = stratification.data_stratified_test

# Load Dataset
housing_train = data_stratified_train.drop(columns=["median_house_value"])
housing_train_labels = data_stratified_train["median_house_value"].copy()
housing_test = data_stratified_test.drop(columns=["median_house_value"])
housing_test_labels = data_stratified_test["median_house_value"].copy()

# Paths
FILE_DIR = os.path.abspath(os.path.dirname(__file__)) 
PARENT_DIR = os.path.dirname(FILE_DIR)
MODEL_DIR = os.path.join(PARENT_DIR, "models")
HOUSING_MODEL_DIR = os.path.join(MODEL_DIR, "housing")
os.makedirs(HOUSING_MODEL_DIR, exist_ok=True)

# ML Algorithms
lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor()

# Standard Deviation (spread of the predicted values): average distance of the predicted values from their own mean.
# RMSE (average deviation error of the predicted values): average value of the differences between the predicted values from their corresponding label values.
# ML Model = ML Algorithm + Dataset
# K-fold cross_val_score: returns the variance (deviation from the label values) of the predicted column values from a ML Algorithm applied on K subsets (folds).
# train_models: saves/loads the model + creates the prediction column based on the ML algorithm + calculates the RMSE 
# of the prediction column values in relation to the label column values for 1 transformed dataset and for 10 subsets of the transformed dataset.
# fine_tune_model: finds the best parameters for a ML Model to have the best performance and minimum RMSE across the dataset and subsets.
# Random Forest: an ensemble algorithm consisting of multiple decision trees
# Decision Trees: built by recursively splitting the dataset into subsets of columns and rows at each node
# n_estimators: number of decision trees in the random forest
# max_features: maximum number of columns randomly selected from the dataset's total columns to consider for each split (node) in the decision trees
# bootstrap=True: rows for each decision tree are randomly selected **with replacement** (default behavior)
# bootstrap=False: all rows in the dataset are used to train each decision tree (no replacement)
# Each text category that is encoded into a binary vector using LabelBinarizer (the text_encoder) 
# becomes its own independent column in the dataset, and the model assigns a separate weight (importance) to each column.

def display_scores(scores):
    print("RMSE for each subset:", scores)
    print("Mean of the RMSEs:", scores.mean())
    print("Standard deviation of the RMSEs:", scores.std())

def train_models(dataset_transformed, dataset_labels, train_models: bool):

    if (train_models == True):

        # Linear Regression
        lin_reg.fit(dataset_transformed, dataset_labels)
        lin_reg_path = os.path.join(HOUSING_MODEL_DIR, "linear_regression_model.pkl")
        joblib.dump(lin_reg, lin_reg_path)
        housing_predictions_lin = lin_reg.predict(dataset_transformed)
        lin_rmse = np.sqrt(mean_squared_error(dataset_labels, housing_predictions_lin))
        lin_scores = cross_val_score(lin_reg, dataset_transformed, dataset_labels, scoring="neg_mean_squared_error", cv=10)
        lin_scores_rmse = np.sqrt(-lin_scores)

        # Decision Tree
        tree_reg.fit(dataset_transformed, dataset_labels)
        tree_reg_path = os.path.join(HOUSING_MODEL_DIR, "decision_tree_model.pkl")
        joblib.dump(tree_reg, tree_reg_path)
        housing_predictions_tree = tree_reg.predict(dataset_transformed)
        tree_rmse = np.sqrt(mean_squared_error(dataset_labels, housing_predictions_tree))
        tree_scores = cross_val_score(tree_reg, dataset_transformed, dataset_labels, scoring="neg_mean_squared_error", cv=10)
        tree_scores_rmse = np.sqrt(-tree_scores)

        # Random Forest
        forest_reg.fit(dataset_transformed, dataset_labels)
        forest_reg_path = os.path.join(HOUSING_MODEL_DIR, "random_forest_model.pkl")
        joblib.dump(forest_reg, forest_reg_path)
        housing_predictions_forest = forest_reg.predict(dataset_transformed)
        forest_rmse = np.sqrt(mean_squared_error(dataset_labels, housing_predictions_forest))
        forest_scores = cross_val_score(forest_reg, dataset_transformed, dataset_labels, scoring="neg_mean_squared_error", cv=10)
        forest_scores_rmse = np.sqrt(-forest_scores)
    else:

        lin_reg_path = os.path.join(HOUSING_MODEL_DIR, "linear_regression_model.pkl")
        tree_reg_path = os.path.join(HOUSING_MODEL_DIR, "decision_tree_model.pkl")
        forest_reg_path = os.path.join(HOUSING_MODEL_DIR, "random_forest_model.pkl")

        # Load pre-trained models
        lin_reg_loaded = joblib.load(lin_reg_path)
        tree_reg_loaded = joblib.load(tree_reg_path)
        forest_reg_loaded = joblib.load(forest_reg_path)

        housing_predictions_lin = lin_reg_loaded.predict(dataset_transformed)
        housing_predictions_tree = tree_reg_loaded.predict(dataset_transformed)
        housing_predictions_forest = forest_reg_loaded.predict(dataset_transformed)

        lin_rmse = np.sqrt(mean_squared_error(dataset_labels, housing_predictions_lin))
        lin_scores = cross_val_score(lin_reg_loaded, dataset_transformed, dataset_labels, scoring="neg_mean_squared_error", cv=10)
        lin_scores_rmse = np.sqrt(-lin_scores)

        tree_rmse = np.sqrt(mean_squared_error(dataset_labels, housing_predictions_tree))
        tree_scores = cross_val_score(tree_reg_loaded, dataset_transformed, dataset_labels, scoring="neg_mean_squared_error", cv=10)
        tree_scores_rmse = np.sqrt(-tree_scores)

        forest_rmse = np.sqrt(mean_squared_error(dataset_labels, housing_predictions_forest))
        forest_scores = cross_val_score(forest_reg_loaded, dataset_transformed, dataset_labels, scoring="neg_mean_squared_error", cv=10)
        forest_scores_rmse = np.sqrt(-forest_scores)
    
    print("Linear Regression (RMSE of the transformed stratified dataset): ", lin_rmse)
    print("Linear Regression (RMSEs for 10 subsets):")
    display_scores(lin_scores_rmse)
    print("\n")

    print("Decision Tree (RMSE of the transformed stratified dataset): ", tree_rmse)
    print("Decision Tree (RMSEs for 10 subsets):")
    display_scores(tree_scores_rmse)
    print("\n")

    print("Random Forest (RMSE of the transformed stratified dataset): ", forest_rmse)
    print("Random Forest (RMSEs for 10 subsets):")
    display_scores(forest_scores_rmse)
    print("\n")

def fine_tune_model(dataset_transformed, dataset_labels, dataset_numerical, dataset_text, model, save_model: bool, model_name="model"):

    # Search for the best parameters for the model: minimum RMSE and best performance across the dataset and subsets
    # Grid of parameters
    param_grid = [
        {"n_estimators":[3,10,30], "max_features":[2,4,6,8]},
        {"bootstrap":[False], "n_estimators":[3,10], "max_features":[2,3,4]}
    ]
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(dataset_transformed, dataset_labels)

    best_parameters = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_model_predictions = best_model.predict(dataset_transformed)
    best_model_rmse = np.sqrt(mean_squared_error(dataset_labels, best_model_predictions))

    cv_results = grid_search.cv_results_
    print("Mean of the RMSEs of the subsets for each parameter combination:")
    for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
        print("Mean of the RMSEs: ", np.sqrt(-mean_score), "for parameters: ", params)
    
    print("Best parameters for the model: ", best_parameters)
    print("Best model RMSE: ", best_model_rmse)

    # Calculates the importance of each column and text category in the formation of the prediction column
    # Retrieve the feature importances (column weights) and combine with column names
    # The order matches because the columns and computed weights are in the same order of the columns fed to the transformation pipeline
    column_weights = [float(weight) for weight in best_model.feature_importances_]
    numerical_columns = list(dataset_numerical)
    new_columns = ["rooms_per_household", "population_per_household"]
    text_encoder = LabelBinarizer()
    text_encoder.fit_transform(dataset_text)
    text_categories = [str(text_category) for text_category in text_encoder.classes_]
    full_columns = numerical_columns + new_columns + text_categories
    sorted_column_weights = sorted(zip(column_weights, full_columns), reverse=True)
    print("Column Weights (Sorted by Importance):")
    print(sorted_column_weights)

    # Test the best model on the test dataset
    dataset_test = housing_test
    dataset_test_labels = housing_test_labels
    dataset_test_transformed = full_pipeline.transform(dataset_test)
    best_model_predictions_test = best_model.predict(dataset_test_transformed)
    best_model_rmse_test = np.sqrt(mean_squared_error(dataset_test_labels, best_model_predictions_test))
    print("Best model RMSE on the test dataset: ", best_model_rmse_test)

    if (save_model):
        model_full_name = model_name + "_fine_tuned_model" + ".pkl"
        fine_tuned_model_path = os.path.join(HOUSING_MODEL_DIR, model_full_name)
        joblib.dump(best_model, fine_tuned_model_path)
        print("Fine-tuned model saved at:", fine_tuned_model_path)

    return best_model

def prediction_columns(dataset_transformed, dataset_labels):

    lin_reg.fit(dataset_transformed, dataset_labels)
    tree_reg.fit(dataset_transformed, dataset_labels)
    forest_reg.fit(dataset_transformed, dataset_labels)
    
    dataset_predictions_lin = lin_reg.predict(dataset_transformed)
    dataset_predictions_tree = tree_reg.predict(dataset_transformed)
    dataset_predictions_forest = forest_reg.predict(dataset_transformed)

    print("Columns of Predictions (Linear Regression x Decision Tree x Random Forest):")
    print(dataset_predictions_lin[0:5], "\n")
    print(dataset_predictions_tree[0:5], "\n")
    print(dataset_predictions_forest[0:5], "\n")
    print("Column of Labels:")
    print(dataset_labels.iloc[0:5].values, "\n")

if __name__ == "__main__":

    dataset_sample = housing_train
    dataset_sample_labels = housing_train_labels
    dataset_sample_numerical = dataset_sample.drop(columns=["ocean_proximity"])
    dataset_sample_text = dataset_sample["ocean_proximity"]
    full_pipeline.fit(dataset_sample)
    dataset_sample_transformed = full_pipeline.transform(dataset_sample)

    prediction_columns(dataset_sample_transformed, dataset_sample_labels)

    train_models(dataset_sample_transformed, dataset_sample_labels, False)

    best_model_forest_reg = fine_tune_model(dataset_sample_transformed, dataset_sample_labels, dataset_sample_numerical, dataset_sample_text, forest_reg, False, "forest_reg")