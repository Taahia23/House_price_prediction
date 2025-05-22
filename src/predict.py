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



# Evaluate the Model's Performance
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_test, y_prediction)
print(f'Root Mean Squared Error: ${rmse:,.2f}')

'''
RMSE measures the average error in dollar amounts
Lower RMSE means better model performance
Example: RMSE of 50,000 means predictions are typically $50,000 off
'''

# Display a Random Sample from the Test Data with Prediction
sample = test_data.sample(1)
sample_features = sample.drop('House_Price', axis=1)
actual_price = sample['House_Price'].values[0]
predicted_price = model.predict(sample_features)[0]

print("\nRandom Sample Evaluation:")
print(f"Actual House Price: ${actual_price:,.2f}")
print(f"Predicted House Price: ${predicted_price:,.2f}")
print(f"Difference: ${abs(actual_price - predicted_price):,.2f}")
print("\nSample Features:")
print(sample_features.to_dict(orient='records')[0])