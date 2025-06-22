import streamlit as st
import joblib
import pandas as pd
import os
from pathlib import Path

st.set_page_config(page_title="Simple House Price Predictor", layout="centered")
st.title("Simple House Price Predictor")


# Load your trained model
@st.cache_data
def load_model():
    try:
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent.parent / 'models' / 'model_with_pipeline.pkl',  # For local development
            Path(__file__).parent / 'model_with_pipeline.pkl',  # If model is copied to app directory
            Path('models/model_with_pipeline.pkl'),  # Relative path
            Path('/mount/src/house_price_prediction/models/model_with_pipeline.pkl')  # For Streamlit Cloud
        ]
        
        for path in possible_paths:
            if path.exists():
                st.write(f"Found model at: {path}")
                return joblib.load(path)
        
        # If we get here, no path worked
        available_files = list(Path(__file__).parent.glob('**/*'))
        st.error(f"Model not found. Available files: {available_files}")
        return None
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

if model is None:
    st.error("""
    ⚠️ Could not load the prediction model. 
    Please check that 'model_with_pipeline.pkl' exists in the correct location.
    """)
    st.stop()

# Rest of your original code remains exactly the same...
st.header("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("Square Footage", min_value=500, value=1500)
    bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, value=2)

with col2:
    year_built = st.number_input("Year Built", min_value=1900, max_value=2023, value=2000)
    lot_size = st.number_input("Lot Size (acres)", min_value=0.1, value=0.5)
    neighborhood = st.slider("Neighborhood Quality (1-10)", 1, 10, 7)

if st.button("Predict Price", type="primary"):
    input_data = pd.DataFrame([{
        'Square_Footage': sqft,
        'Num_Bedrooms': bedrooms,
        'Num_Bathrooms': bathrooms,
        'Year_Built': year_built,
        'Lot_Size': lot_size,
        'Garage_Size': 1,
        'Neighborhood_Quality': neighborhood
    }])
    
    price = model.predict(input_data)[0]
    st.success(f"### Predicted Price: ${price:,.2f}")
    st.balloons()

st.divider()
st.write("""
**How to use:**
1. Fill in the property details
2. Click "Predict Price"
3. See the estimated value!

*Note: This is a simple estimate based on machine learning.*
""")