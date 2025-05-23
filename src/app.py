import joblib
import pandas as pd

# Create sample data
sample_data = {
    'Square_Footage': 2500,
    'Num_Bedrooms': 3,
    'Num_Bathrooms': 2,
    'Year_Built': 1995,
    'Lot_Size': 0.5,
    'Garage_Size': 1,
    'Neighborhood_Quality': 7
}


# Convert to DataFrame
sample_data_df = pd.DataFrame([sample_data])

# Load the trained model
model = joblib.load('models/model_with_pipeline.pkl')

# Make prediction
result = model.predict(sample_data_df)

print(f"Predicted House Price: ${result[0]:,.2f}")
