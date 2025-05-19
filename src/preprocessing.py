import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
import joblib
import os

# 1. Load the data
data = pd.read_csv('data/house_price_regression_dataset.csv')

print(data.head())

# Check for missing values
print("Missing values in dataset:")
print(data.isnull().sum())

# Categorize the House_Price variable into bins for stratified sampling
data['price_cat'] = pd.cut(data['House_Price'],
                           bins=[0, 200000, 400000, 600000, 800000, np.inf],
                           labels=[1, 2, 3, 4, 5])

# Split data into train and test sets with stratification
train_set, test_set = train_test_split(data,
                                      test_size=0.2,
                                      random_state=42,
                                      stratify=data['price_cat'])

# Drop the temporary price category column
train_set.drop('price_cat', axis=1, inplace=True)
test_set.drop('price_cat', axis=1, inplace=True)

print(f'Train set shape: {train_set.shape}')
print(f'Test set shape: {test_set.shape}')

# Save train and test sets
os.makedirs('data', exist_ok=True)
train_set.to_csv('data/train_set.csv', index=False)
test_set.to_csv('data/test_set.csv', index=False)

# Reload training data
train_set = pd.read_csv('data/train_set.csv')

print("\nData types:")
print(train_set.dtypes)

# Split features and target variable
X_train = train_set.drop('House_Price', axis=1)
y_train = train_set['House_Price'].copy()

# Create a validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                 test_size=0.2,
                                                 random_state=42)

print(f'\nTrain set shape: {X_train.shape}')
print(f'Validation set shape: {X_val.shape}')

# Identify numerical and categorical columns
numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = X_train.select_dtypes('object').columns.tolist()

print(f'\nNumeric columns: {numeric_columns}')
print(f'Categorical columns: {categorical_columns}')

# Preprocessing: Handling Missing Values
print(f'\nMissing values before preprocessing:')
print(X_train.isnull().sum())

# Create imputers and scaler
numeric_imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# For this dataset, there are no categorical columns, so we don't need categorical imputer/encoder

# Preprocess numerical features
X_train[numeric_columns] = numeric_imputer.fit_transform(X_train[numeric_columns])
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])

X_val[numeric_columns] = numeric_imputer.transform(X_val[numeric_columns])
X_val[numeric_columns] = scaler.transform(X_val[numeric_columns])

print(f'\nAfter preprocessing:')
print(X_train.head())

# Train a Linear Regression Model
lin_regression = LinearRegression()
lin_regression.fit(X_train, y_train)
y_predict = lin_regression.predict(X_val)
rmse = root_mean_squared_error(y_val, y_predict)
print(f'\nRMSE of Linear regression: {rmse}')

# Train a Random Forest Model
random_forest = RandomForestRegressor(n_estimators=120, random_state=42)
random_forest.fit(X_train, y_train)
y_predict = random_forest.predict(X_val)
rmse = root_mean_squared_error(y_val, y_predict)
print(f'RMSE of Random Forest: {rmse}')

# Train a Neural Network
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), 
                  activation='relu', 
                  solver='adam', 
                  random_state=42,
                  max_iter=500)
mlp.fit(X_train, y_train)
y_predict = mlp.predict(X_val)
rmse = root_mean_squared_error(y_val, y_predict)
print(f'RMSE of MLP Regressor: {rmse}')

# Save the best model and preprocessors
os.makedirs('models', exist_ok=True)
joblib.dump(mlp, 'models/mlp_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(numeric_imputer, 'models/num_imputer.pkl')

print('\nModels and preprocessors saved successfully.')