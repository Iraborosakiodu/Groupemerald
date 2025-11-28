import streamlit as st
import pandas as pd
import pickle

# -----------------------------------------------------------------------------
# 1. SETUP & LOADING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

@st.cache_resource
def load_model():
    try:
        # Load the model we saved in model.py
        with open('house_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['encoders']
    except FileNotFoundError:
        return None, None

def main():
    st.title("🏠 Real Estate Price Predictor")
    st.write("Enter the house details below to estimate its market value.")

    # Load the trained model
    model, encoders = load_model()

    if not model:
        st.error("Model file not found! Please run 'model.py' first to generate the model.")
        st.stop()

    # -------------------------------------------------------------------------
    # 2. USER INPUT FORM
    # -------------------------------------------------------------------------
    with st.form("house_form"):
        col1, col2 = st.columns(2)
        
        # Numerical Inputs (Area, Beds, Baths, Year)
        with col1:
            area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=2000)
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
            year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2000)

        # Categorical Inputs (Floors, Location, Condition, Garage)
        with col2:
            floors = st.number_input("Floors", min_value=1, max_value=5, value=1)
            
            # Use the encoders to show the correct text options (e.g., "Urban", "Rural")
            loc_options = encoders['Location'].classes_
            location = st.selectbox("Location", loc_options)
            
            cond_options = encoders['Condition'].classes_
            condition = st.selectbox("Condition", cond_options)
            
            garage_options = encoders['Garage'].classes_
            garage = st.selectbox("Has Garage?", garage_options)

        submitted = st.form_submit_button("Predict Price 💰")

    # -------------------------------------------------------------------------
    # 3. PREDICTION LOGIC
    # -------------------------------------------------------------------------
    if submitted:
        # 1. Encode the text inputs to numbers (e.g., "Urban" -> 1)
        loc_encoded = encoders['Location'].transform([location])[0]
        cond_encoded = encoders['Condition'].transform([condition])[0]
        garage_encoded = encoders['Garage'].transform([garage])[0]

        # 2. Organize data exactly as the model expects
        input_data = pd.DataFrame({
            'Area': [area],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Floors': [floors],
            'YearBuilt': [year_built],
            'Location': [loc_encoded],
            'Condition': [cond_encoded],
            'Garage': [garage_encoded]
        })

        # 3. Make Prediction
        predicted_price = model.predict(input_data)[0]

        # 4. Show Result
        st.success(f"### Estimated House Price: ${predicted_price:,.2f}")

if __name__ == "__main__":
    main()
