import joblib
import pandas as pd
import streamlit as st

# Load the trained pipeline
pipeline = joblib.load("california_rf_model.pkl")

st.title("🏡 California Housing Price Prediction")
st.write("Please provide the input features below and click 'Predict' to get the estimated house price.")


# Define function to make predictions
def predict(input_features):
    # Convert the list of features to a DataFrame and set the column names
    input_df = pd.DataFrame([input_features],
                            columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                                     'Latitude', 'Longitude'])

    # Make the prediction using the pipeline
    result = pipeline.predict(input_df)

    return result[0]


# Define the form with columns for better layout
st.header("📋 Input Features")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("💰 Median Income", min_value=0, max_value=100000, value=45000, step=1000,
                       help="Enter the median income in dollars. For example, enter 45000 for an income of $45,000.")
    HouseAge = st.slider("🏠 House Age (in years)", min_value=0, max_value=100, value=20, step=1,
                         help="Enter the age of the house in years.")
    AveRooms = st.slider("🚪 Average Number of Rooms", min_value=1, max_value=20, value=5, step=1,
                         help="Enter the average number of rooms in the house.")
    AveBedrms = st.slider("🛏️ Average Number of Bedrooms", min_value=1, max_value=10, value=2, step=1,
                          help="Enter the average number of bedrooms in the house.")

with col2:
    Population = st.slider("👨‍👩‍👧‍👦 Population", min_value=0, max_value=10000, value=2000, step=100,
                           help="Enter the population of the neighborhood.")
    AveOccup = st.slider("🏠 Average Occupancy", min_value=0, max_value=10, value=3, step=1,
                         help="Enter the average occupancy of the house.")
    Latitude = st.slider("🌍 Latitude", min_value=32.54, max_value=42.00, value=37.00, step=0.01,
                         help="Enter the latitude of the house location.")
    Longitude = st.slider("🌍 Longitude", min_value=-124.3, max_value=-114.3, value=-120.0, step=0.01,
                          help="Enter the longitude of the house location.")

# Collecting all inputs
features = [MedInc / 1000, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]

# Button for making predictions
if st.button("🔍 Predict"):
    prediction = predict(features)
    st.subheader(f"🏡 Predicted House Price: ${prediction * 100000:,.2f}")
    st.write("Please note that the predicted price is an estimate and may vary.")

st.write("Adjust the input features and click 'Predict' again to get a new estimated house price.")
