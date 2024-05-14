# data_loader.py
import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y
