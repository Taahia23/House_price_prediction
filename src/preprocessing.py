import pandas as pd
import numpy as np
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/house_price_regression_dataset.csv");

df = df.dropna()
X=df.drop("House_Price", axis=1)
y=df["House_Price"]
