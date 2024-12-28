# transformation_pipeline.py
"""
This module contains the full transformation pipeline
for preprocessing numerical and text columns from a dataset.
"""
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

# Modules
import data_download
import data_display
import stratification
import transformer
import transformation_pipeline

# References
custom_dataframeselector = transformer.CustomDataFrameSelector
custom_labelbinarizer = transformer.CustomLabelBinarizer
custom_transformer = transformer.CustomTransformer

# Deletion of the Label Column and separation of the Text Column
housing = stratification.data_stratified_train.drop(columns=["median_house_value"])
housing_numerical = housing.drop(columns=["ocean_proximity"])
housing_text = housing["ocean_proximity"]

# Dataset Columns to be processed by the Pipeline
# Each text category, encoded into a binary vector, is treated as an independent column with some specific weight (importance) for the creation of the prediction column
numerical_columns = list(housing_numerical)
text_columns = ["ocean_proximity"] 
text_categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

# Tranformation Pipeline
numerical_pipeline = Pipeline([
    ("selector", custom_dataframeselector(numerical_columns)),
    ("imputer", SimpleImputer(strategy="median")),              
    ("transformer", custom_transformer()),                       
    ("std_scaler", StandardScaler())                            
])

text_pipeline = Pipeline([
    ("selector", custom_dataframeselector(text_columns)),
    ("label_binarizer", custom_labelbinarizer(text_categories))                
])

full_pipeline = FeatureUnion(transformer_list=[
    ("numerical_pipeline", numerical_pipeline),
    ("text_pipeline", text_pipeline)
])

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

if __name__ == "__main__":

    # Transform the Dataset
    transformer_function = transformer.CustomTransformer()
    housing_transformed = full_pipeline.fit_transform(housing)
    housing_transformed_df = transform_dataframe(housing_transformed, housing_numerical, housing_text, ["rooms_per_household", "population_per_household"])

    # Dataset
    print("Dataset:")
    print(housing.iloc[0:1], "\n")

    # Dataset Transformed
    print("Dataset Transformed:")
    print(housing_transformed[0:1], "\n")

    print("Dataframe Transformed:")
    print(housing_transformed_df.iloc[0:1], "\n")