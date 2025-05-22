import joblib
import pandas as pd

model = joblib.load('models/model_with_pipeline.pkl')  # or 'models/mlp_model.pkl'


# Load the Test Data
test_data = pd.read_csv('data/test_set.csv')

# Separate Features (X_test) and Target (y_test)
'''
X_test contains all input features (independent variables)
y_test contains the actual house prices (dependent variable)
'''
X_test = test_data.drop('House_Price', axis=1)
y_test = test_data['House_Price'].copy()


# Make Predictions Using the Model
y_prediction = model.predict(X_test)
