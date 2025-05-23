import streamlit as st
import joblib
import pandas as pd

# Set up the app
st.set_page_config(page_title="Simple House Price Predictor", layout="centered")
st.title("Simple House Price Predictor")


# Load your trained model
@st.cache_data
def load_model():
    return joblib.load('models/model_with_pipeline.pkl')

model = load_model()


# Input fields
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
    
    # Show result
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