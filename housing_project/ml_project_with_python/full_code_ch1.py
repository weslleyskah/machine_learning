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

# Display options
pd.set_option("display.max_columns", None)                           
pd.set_option("display.width", shutil.get_terminal_size().columns)

# Download or Load the Dataset
def get_data(data_download: bool, data_load: bool):

    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(FILE_DIR)
    DATA_PATH = os.path.join(PARENT_DIR, "datasets", "housing")
    DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"

    if (data_download):
        if not os.path.isdir(DATA_PATH):
            os.makedirs(DATA_PATH)
        tgz_path = os.path.join(DATA_PATH, "housing.tgz")    
        urllib.request.urlretrieve(DATA_URL, tgz_path)
        data_tgz = tarfile.open(tgz_path)
        data_tgz.extractall(path=DATA_PATH)
        data_tgz.close()
    
    if (data_load):
        csv_path = os.path.join(DATA_PATH, "housing.csv")
        return pd.read_csv(csv_path)

# Transform the dataset into a dataframe
def transform_dataframe(dataset_transformed, dataset_numerical = None, dataset_text = None, new_numerical_columns = None, new_text_columns = None):

    full_columns = []

    if (dataset_numerical is not None):
        numerical_columns = list(dataset_numerical.columns)
        full_columns += numerical_columns
    if (new_numerical_columns is not None):
        full_columns += new_numerical_columns
    if (dataset_text is not None):
        text_encoder = LabelBinarizer()
        text_encoder.fit_transform(dataset_text)
        text_categories = [str(text_category) for text_category in text_encoder.classes_]
        full_columns += text_categories
    if (new_text_columns is not None):
        full_columns += new_text_columns
    
    dataframe_transformed = pd.DataFrame(dataset_transformed, columns=full_columns)

    return dataframe_transformed

# Stratification of the Dataset
def stratify_dataset(dataset):
    # 1: Add Stratification Column: The stratification column is based on a dataset column
    # It scales down the the median_income values (decimals x$10,000) and limits them to 5 values on the income-category column, representing 5 ranges for the median_income values
    dataset["income-category"] = np.ceil(dataset["median_income"] / 1.5)
    dataset["income-category"].where(dataset["income-category"] < 5, 5.0)

    # 2: Stratification Algorithm
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(dataset, dataset["income-category"]):
        dataset_stratified_train = dataset.loc[train_index]
        dataset_stratified_test = dataset.loc[test_index]

    # 3: Clean up: Remove stratification column from all the datasets
    for set_ in (dataset, dataset_stratified_train, dataset_stratified_test):
        set_.drop("income-category", axis=1, inplace=True)
    
    return dataset_stratified_train, dataset_stratified_test

# Transformation Pipeline
def transformation_pipeline():

    # Selector: select the numerical and text columns from a dataset
    class CustomDataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, columns):
            self.columns = columns
        def fit(self, dataset, dataset_label = None):
            return self
        def transform(self, dataset, dataset_label = None):
            return dataset[self.columns].values

    # Text Encoder: convert text columns into binary vectors
    class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
        def __init__(self, categories = None):
            self.categories = categories
            self.label_binarizer = None

        def fit(self, dataset, dataset_label = None):
            if self.categories is not None:
                self.label_binarizer = LabelBinarizer()
                self.label_binarizer.fit(self.categories)
            else:
                self.label_binarizer = LabelBinarizer()
                self.label_binarizer.fit(dataset)
            return self

        def transform(self, dataset, dataset_label = None):
            return self.label_binarizer.transform(dataset)

    # Custom Transformer Class: group of functions to modify the dataset
    class CustomTransformer(BaseEstimator, TransformerMixin):

        def __init__(self):
            pass
        
        def fit(self, dataset, dataset_label = None):
            return self

        # Creation of new columns
        def transform(self, dataset, dataset_label = None):

            rooms_index, population_index, households_index = 3, 5, 6
            rooms_per_household = dataset[:, rooms_index] / dataset[:, households_index] 
            population_per_household = dataset[:, population_index] / dataset[:, households_index]

            return np.c_[dataset, rooms_per_household, population_per_household]
    
    # Dataset Columns to be processed by the Pipeline
    numerical_columns = list(housing_stratified_train.drop(columns=["ocean_proximity"]))
    text_columns = ["ocean_proximity"] 
    text_categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

    # Tranformation Pipeline
    numerical_pipeline = Pipeline([
        ("selector", CustomDataFrameSelector(numerical_columns)),
        ("imputer", SimpleImputer(strategy="median")),              
        ("transformer", CustomTransformer()),                       
        ("std_scaler", StandardScaler())                            
    ])

    text_pipeline = Pipeline([
        ("selector", CustomDataFrameSelector(text_columns)),
        ("label_binarizer", CustomLabelBinarizer(text_categories))                
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("numerical_pipeline", numerical_pipeline),
        ("text_pipeline", text_pipeline)
    ])

    return full_pipeline

# References
data = get_data(True, True)
full_pipeline = transformation_pipeline()

# Housing Dataset
housing = data.copy()
housing_stratified_train, housing_stratified_test = stratify_dataset(housing)
housing_train = housing_stratified_train.drop(columns=["median_house_value"]).copy()
housing_train_labels = housing_stratified_train["median_house_value"].copy()
housing_test = housing_stratified_test.drop(columns=["median_house_value"]).copy()
housing_test_labels = housing_stratified_test["median_house_value"].copy()

# ML Algorithms
lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor()

def display_scores(scores):
    print("RMSE for each subset:", scores)
    print("Mean of the RMSEs:", scores.mean())
    print("Standard deviation of the RMSEs:", scores.std())

def train_models(dataset_transformed, dataset_labels, save_models: bool):

    FILE_DIR = os.path.abspath(os.path.dirname(__file__)) 
    PARENT_DIR = os.path.dirname(FILE_DIR)
    MODEL_DIR = os.path.join(PARENT_DIR, "models")

    if (save_models == True):

        # Linear Regression
        lin_reg.fit(dataset_transformed, dataset_labels)
        lin_reg_path = os.path.join(MODEL_DIR, "linear_regression_model.pkl")
        joblib.dump(lin_reg, lin_reg_path)
        housing_predictions_lin = lin_reg.predict(dataset_transformed)
        lin_rmse = np.sqrt(mean_squared_error(dataset_labels, housing_predictions_lin))
        lin_scores = cross_val_score(lin_reg, dataset_transformed, dataset_labels, scoring="neg_mean_squared_error", cv=10)
        lin_scores_rmse = np.sqrt(-lin_scores)

        # Decision Tree
        tree_reg.fit(dataset_transformed, dataset_labels)
        tree_reg_path = os.path.join(MODEL_DIR, "decision_tree_model.pkl")
        joblib.dump(tree_reg, tree_reg_path)
        housing_predictions_tree = tree_reg.predict(dataset_transformed)
        tree_rmse = np.sqrt(mean_squared_error(dataset_labels, housing_predictions_tree))
        tree_scores = cross_val_score(tree_reg, dataset_transformed, dataset_labels, scoring="neg_mean_squared_error", cv=10)
        tree_scores_rmse = np.sqrt(-tree_scores)

        # Random Forest
        forest_reg.fit(dataset_transformed, dataset_labels)
        forest_reg_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
        joblib.dump(forest_reg, forest_reg_path)
        housing_predictions_forest = forest_reg.predict(dataset_transformed)
        forest_rmse = np.sqrt(mean_squared_error(dataset_labels, housing_predictions_forest))
        forest_scores = cross_val_score(forest_reg, dataset_transformed, dataset_labels, scoring="neg_mean_squared_error", cv=10)
        forest_scores_rmse = np.sqrt(-forest_scores)
    else:
        lin_reg_path = os.path.join(MODEL_DIR, "linear_regression_model.pkl")
        tree_reg_path = os.path.join(MODEL_DIR, "decision_tree_model.pkl")
        forest_reg_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")

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

    FILE_DIR = os.path.abspath(os.path.dirname(__file__)) 
    PARENT_DIR = os.path.dirname(FILE_DIR)
    MODEL_DIR = os.path.join(PARENT_DIR, "models")

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
        fine_tuned_model_path = os.path.join(MODEL_DIR, model_full_name)
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

    # prediction_columns(dataset_sample_transformed, dataset_sample_labels)

    # train_models(dataset_sample_transformed, dataset_sample_labels, False)

    best_model_forest_reg = fine_tune_model(dataset_sample_transformed, dataset_sample_labels, dataset_sample_numerical, dataset_sample_text, forest_reg, False, "forest_reg")