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

# List of cities with their latitude and longitude
cities = {
    "Los Angeles": (34.0522, -118.2437),
    "San Francisco": (37.7749, -122.4194),
    "San Diego": (32.7157, -117.1611),
    "Sacramento": (38.5816, -121.4944),
    "Fresno": (36.7378, -119.7871),
    "San Jose": (37.3382, -121.8863),
    "Oakland": (37.8044, -122.2711),
    "Long Beach": (33.7701, -118.1937),
    "Santa Ana": (33.7456, -117.8678),
    "Anaheim": (33.8366, -117.9143),
    "Riverside": (33.9806, -117.3755),
    "Stockton": (37.9577, -121.2908),
    "Chula Vista": (32.6401, -117.0842),
    "Irvine": (33.6846, -117.8265),
    "Fremont": (37.5485, -121.9886),
    "San Bernardino": (34.1083, -117.2898),
    "Modesto": (37.6391, -120.9969),
    "Fontana": (34.0922, -117.4351),
    "Oxnard": (34.1975, -119.1771),
    "Moreno Valley": (33.9425, -117.2297),
    "Glendale": (34.1425, -118.2551),
    "Huntington Beach": (33.6603, -117.9992),
    "Santa Clarita": (34.3917, -118.5426),
    "Garden Grove": (33.7743, -117.9374),
    "Oceanside": (33.1959, -117.3795),
    "Rancho Cucamonga": (34.1064, -117.5931),
    "Santa Rosa": (38.4405, -122.7141),
    "Ontario": (34.0633, -117.6509),
    "Elk Grove": (38.4088, -121.3716),
    "Corona": (33.8753, -117.5664),
    "Lancaster": (34.6868, -118.1542),
    "Palmdale": (34.5794, -118.1165),
    "Salinas": (36.6777, -121.6555),
    "Hayward": (37.6688, -122.0808),
    "Pomona": (34.0551, -117.7497),
    "Sunnyvale": (37.3688, -122.0363),
    "Escondido": (33.1192, -117.0864),
    "Torrance": (33.8358, -118.3406),
    "Pasadena": (34.1478, -118.1445),
    "Orange": (33.7879, -117.8531),
    "Fullerton": (33.8704, -117.9242),
    "Roseville": (38.7521, -121.288),
    "Visalia": (36.3302, -119.2921),
    "Thousand Oaks": (34.1706, -118.8376),
    "Concord": (37.9779, -122.0311),
    "Simi Valley": (34.2694, -118.7815),
    "Santa Clara": (37.3541, -121.9552),
    "Victorville": (34.5361, -117.2912),
    "Vallejo": (38.1041, -122.2566),
    "Berkeley": (37.8715, -122.273),
    "El Monte": (34.0686, -118.0276),
    "Downey": (33.9401, -118.1332),
    "Costa Mesa": (33.6638, -117.9033),
    "Carlsbad": (33.1581, -117.3506),
    "Fairfield": (38.2494, -122.040),
    "Temecula": (33.4936, -117.1484),
    "Inglewood": (33.9617, -118.3531),
    "Antioch": (38.0049, -121.8058),
    "Murrieta": (33.5539, -117.2139),
    "Richmond": (37.9358, -122.3477),
    "Ventura": (34.2746, -119.229),
    "West Covina": (34.0686, -117.938),
    "Norwalk": (33.9022, -118.0817),
    "Daly City": (37.6879, -122.4702),
    "Burbank": (34.1808, -118.3089),
    "Santa Maria": (34.953, -120.4357),
    "Clovis": (36.8252, -119.7029),
    "El Cajon": (32.7948, -116.9625),
    "San Mateo": (37.5629, -122.3255),
    "Rialto": (34.1118, -117.3883),
    "Vista": (33.2000, -117.2425),
    "Jurupa Valley": (33.9972, -117.4855),
    "Compton": (33.8958, -118.2201),
    "Mission Viejo": (33.5961, -117.6594),
    "South Gate": (33.9547, -118.212),
    "Carson": (33.8314, -118.282),
    "Santa Monica": (34.0195, -118.4912),
    "Hesperia": (34.4264, -117.3009),
    "Westminster": (33.7513, -117.9939),
    "Redding": (40.5865, -122.3917),
    "Santa Barbara": (34.4208, -119.6982),
    "Chico": (39.7285, -121.8375),
    "Whittier": (33.9792, -118.0328),
    "Newport Beach": (33.6189, -117.9289),
    "Hawthorne": (33.9164, -118.3526),
    "San Marcos": (33.1434, -117.1661),
    "Citrus Heights": (38.7071, -121.2811),
    "Alhambra": (34.0953, -118.127),
    "Tracy": (37.7397, -121.4252),
    "Livermore": (37.6819, -121.768),
    "Buena Park": (33.8675, -117.9981),
    "Lakewood": (33.8536, -118.1332),
    "Merced": (37.3022, -120.481),
    "Hemet": (33.7475, -116.9718),
    "Chino": (34.0122, -117.6889),
    "Menifee": (33.6783, -117.1661),
    "Lake Forest": (33.6469, -117.6892),
    "Napa": (38.5025, -122.2654),
    "Redwood City": (37.4852, -122.2364),
    "Bellflower": (33.8817, -118.117),
    "Indio": (33.7206, -116.2156),
    "Tustin": (33.742, -117.8236),
    "Baldwin Park": (34.0853, -117.9609),
    "Chino Hills": (33.9894, -117.7326),
    "Mountain View": (37.3861, -122.0838),
    "Alameda": (37.7652, -122.2416),
    "Upland": (34.0975, -117.6484),
    "Folsom": (38.6779, -121.1761),
    "San Ramon": (37.7799, -121.978),
    "Pleasanton": (37.6624, -121.8747),
    "Lynwood": (33.9303, -118.2115),
    "Union City": (37.5934, -122.0438),
    "Apple Valley": (34.5008, -117.1859),
    "Redlands": (34.0556, -117.1825),
    "Turlock": (37.4947, -120.8466),
    "Perris": (33.7825, -117.2286),
    "Manteca": (37.7974, -121.2161),
    "Milpitas": (37.4323, -121.8996),
    "Redondo Beach": (33.8492, -118.3884),
    "Davis": (38.5449, -121.7405),
    "Camarillo": (34.2164, -119.0376),
    "Yuba City": (39.1404, -121.6169),
    "Rancho Cordova": (38.5891, -121.3027),
    "Palo Alto": (37.4419, -122.143),
    "Yorba Linda": (33.8886, -117.8133),
    "Walnut Creek": (37.9101, -122.0652),
    "Pittsburg": (38.0278, -121.8847),
    "Laguna Niguel": (33.5225, -117.7076),
    "San Leandro": (37.7249, -122.1561),
    "Eureka": (40.8021, -124.1637),
    "Lodi": (38.1302, -121.2724),
    "Woodland": (38.6785, -121.7733),
    "Covina": (34.0900, -117.8903),
    "Montebello": (34.0165, -118.1138),
    "Encinitas": (33.0369, -117.2919),
    "Cupertino": (37.3229, -122.0322),
    "Vacaville": (38.3566, -121.9877),
    "Culver City": (34.0219, -118.3965),
    "Gilroy": (37.0058, -121.5683),
    "Los Gatos": (37.2358, -121.9624),
    "San Clemente": (33.4269, -117.611),
    "La Habra": (33.9319, -117.9462),
    "Monterey Park": (34.0625, -118.1228),
    "La Mesa": (32.7678, -117.0231),
    "Arcadia": (34.1397, -118.0353),
    "Poway": (32.9628, -117.0359),
    "Rowland Heights": (33.9761, -117.9053),
    "Azusa": (34.1336, -117.9076),
    "Cathedral City": (33.7805, -116.4668),
    "San Gabriel": (34.0961, -118.1058),
    "Cypress": (33.816, -118.0373),
    "Ceres": (37.5949, -120.9577),
    "Rohnert Park": (38.3396, -122.7011),
    "Yucaipa": (34.0336, -117.0431),
    "Laguna Beach": (33.5427, -117.7854),
    "Encino": (34.1592, -118.5012),
}

# Sidebar input
st.sidebar.title("Input Features")
city = st.sidebar.selectbox("Select City", list(cities.keys()))
MedInc = st.sidebar.slider("Median Income (scaled by 1000)", 0.5, 15.0, 3.0)
HouseAge = st.sidebar.slider("House Age", 1, 50, 20)
AveRooms = st.sidebar.slider("Average Rooms", 2.0, 10.0, 5.0)
AveBedrms = st.sidebar.slider("Average Bedrooms", 1.0, 5.0, 2.0)
Population = st.sidebar.slider("Population", 100.0, 30000.0, 1500.0)
AveOccup = st.sidebar.slider("Average Occupancy", 1.0, 6.0, 3.0)

# Get the latitude and longitude of the selected city
latitude, longitude = cities[city]

# Display inputs
st.subheader("Selected Inputs")
st.write(f"**City**: {city}")
st.write(f"**Latitude**: {latitude}")
st.write(f"**Longitude**: {longitude}")
st.write(f"**Median Income (scaled by 1000)**: {MedInc}")
st.write(f"**House Age**: {HouseAge}")
st.write(f"**Average Rooms**: {AveRooms}")
st.write(f"**Average Bedrooms**: {AveBedrms}")
st.write(f"**Population**: {Population}")
st.write(f"**Average Occupancy**: {AveOccup}")

# Button to trigger prediction
if st.button("Predict"):
    features = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, latitude, longitude]
    prediction = predict(features)
    # Assuming the prediction result is in the same scale as training data (typically, tens or hundreds of thousands)
    st.subheader("Predicted House Price")
    st.write(f"The estimated house price is: ${prediction * 1000:,.2f}")
