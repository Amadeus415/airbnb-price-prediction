import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('./datasets/airbnb_cleaned.csv')

df = load_data()

# Filter rows with valid coordinates
df = df.dropna(subset=['coordinates/latitude', 'coordinates/longitude'])

# Create the map centered at the average coordinates of the listings
center_lat = df['coordinates/latitude'].mean()
center_long = df['coordinates/longitude'].mean()

# Create the folium map
m = folium.Map(location=[center_lat, center_long], zoom_start=12)

# Add markers to the map for each listing
for index, row in df.iterrows():
    folium.Marker(
        location=[row['coordinates/latitude'], row['coordinates/longitude']],
        popup=f"Price: {row['price_per_night']} per night\nGuests: {row['max_guests']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# Display the map in Streamlit
st.title('Airbnb Listings Map')
st.write('This map shows the location of Airbnb listings in San Rafael')
st_folium(m, width=725)

#-----------------------Price Prediction--------------------------------------------

# Prepare the data for prediction
# We'll use max_guests, Beds, and Baths as features and price_per_night as the target
df = df.dropna(subset=['max_guests', 'Beds', 'Baths', 'price_per_night', 'location'])

# One-hot encode the 'location' column
encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' to avoid multicollinearity
location_encoded = encoder.fit_transform(df[['location']])

# Convert the one-hot encoded array to a DataFrame and concatenate with the original DataFrame
location_encoded_df = pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out(['location']))
df_encoded = pd.concat([df, location_encoded_df], axis=1)

# Features: max_guests, Beds, Baths, and the one-hot encoded locations
X = df_encoded[['max_guests', 'Beds', 'Baths'] + list(location_encoded_df.columns)]
y = df_encoded['price_per_night']


# Train a simple linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Create a form for user input
st.title('🏡 Airbnb Price Estimator')

with st.form("price_estimation_form"):
    address = st.text_input("🏠 Home Address")
    max_guests = st.number_input("👥 Maximum Guests", min_value=1, step=1)
    beds = st.number_input("🛏️ Number of Beds", min_value=1, step=1)
    baths = st.number_input("🛁 Number of Baths", min_value=1.0, step=0.5)
    location = st.selectbox(
        "📍 Select Location",
        ["San Rafael", "Novato", "Fairfax", "Sausalito", "Mill Valley", "Kentfield", "Stinson Beach"]
    )
    
    # Submit button
    submit_button = st.form_submit_button(label="Estimate Price")


# Predict the price when the user submits the form
if submit_button:
    if address:  # Ensure the address field is filled
        # One-hot encode the selected location
        location_encoded_user = encoder.transform([[location]])
        location_encoded_df_user = pd.DataFrame(location_encoded_user, columns=encoder.get_feature_names_out(['location']))

        # Create a DataFrame for user input
        user_input = pd.DataFrame({
            'max_guests': [max_guests],
            'Beds': [beds],
            'Baths': [baths]
        })
        
        # Add the one-hot encoded location columns to the user input DataFrame
        user_input = pd.concat([user_input, location_encoded_df_user], axis=1)

        # Ensure all columns match the training set (important if new locations are added later)
        missing_cols = set(X.columns) - set(user_input.columns)
        for col in missing_cols:
            user_input[col] = 0

        # Ensure the columns are in the same order as the training data
        user_input = user_input[X.columns]
        
        # Make the prediction
        predicted_price = model.predict(user_input)[0]
        
        # Display the result in a visually appealing way
        st.success(f"🏡 Estimated Price per Night for {address} in {location}:")
        st.metric(label="💰 Estimated Price", value=f"${predicted_price:.2f}")

        st.success(f"🏡 Estimated Price per Week for {address} in {location}:")
        st.metric(label="💰 Estimated Price", value=f"${predicted_price*7:.2f}")


        st.write("For more details, please contact your local Airbnb representative.")
    else:
        st.warning("Please provide a valid address!")