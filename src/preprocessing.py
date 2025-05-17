import pandas as pd
import numpy as np
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filepath):
    filepath = "../data/house_price_regression_dataset.csv"
    return pd.read_csv(filepath)