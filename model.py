import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
import requests

# Load Data
@st.cache
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
        return None

# Fetch Live Match Data
def fetch_live_data(api_url):
    st.write("### Fetching Live Data")
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            live_data = response.json()
            st.success("Live data fetched successfully.")
            return pd.DataFrame(live_data)
        else:
            st.error(f"Failed to fetch live data. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred while fetching live data: {e}")
        return None

# Data Exploration and Visualization
def explore_data(data):
    st.write("### Dataset Overview")
    st.write(data.head())
    
    st.write("### Data Summary")
    st.write(data.describe())

    st.write("### Missing Values")
    st.write(data.isnull().sum())

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Build Prediction Model
def train_model(data):
    st.write("### Training Prediction Model")

    # Feature Selection (Assuming relevant columns exist)
    features = [
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bonus', 'influence', 'creativity', 'threat'
    ]
    target = 'total_points'

    if not all(col in data.columns for col in features + [target]):
        st.error("Required columns for training are missing in the dataset.")
        return None

    X = data[features]
    y = data[target]

    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")

    return model

# Make Predictions
def make_predictions(model, data):
    st.write("### Make Predictions")

    features = [
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bonus', 'influence', 'creativity', 'threat'
    ]

    if not all(col in data.columns for col in features):
        st.error("Required columns for predictions are missing in the dataset.")
        return

    X = data[features]
    X = X.fillna(0)

    predictions = model.predict(X)
    data['predicted_points'] = predictions

    st.write("### Predicted Points")
    st.write(data[['name', 'predicted_points']].sort_values(by='predicted_points', ascending=False).head(10))

    # Visualization
    st.write("### Predicted Points Distribution")
    fig = px.histogram(data, x='predicted_points', nbins=20, title='Distribution of Predicted Points')
    st.plotly_chart(fig)

    return data

# AI-Driven Transfer Suggestions
def suggest_transfers(data):
    st.write("### AI-Driven Transfer Suggestions")

    # Suggest players to transfer in based on predicted points
    transfer_in_candidates = data[['name', 'predicted_points']].sort_values(by='predicted_points', ascending=False).head(5)
    st.write("#### Top Players to Transfer In")
    st.write(transfer_in_candidates)

    # Example logic for players to transfer out
    if 'form' in data.columns:
        transfer_out_candidates = data[data['form'] < 3][['name', 'form']].sort_values(by='form').head(5)
        st.write("#### Players to Consider Transferring Out")
        st.write(transfer_out_candidates)
    else:
        st.write("Form data not available. Cannot suggest players to transfer out.")

# Track Opponent Behavior
def track_opponents(data):
    st.write("### Opponent Tracking")
    
    # Example Analysis: Chips Usage
    if 'chip_usage' in data.columns:
        chip_counts = data['chip_usage'].value_counts()
        st.write("#### Chip Usage Summary")
        st.bar_chart(chip_counts)

    if 'budget' in data.columns:
        st.write("#### Budget Flexibility")
        fig = px.box(data, x='budget', title='Distribution of Remaining Budgets')
        st.plotly_chart(fig)

# Streamlit App Main Function
def main():
    st.title("Fantasy Premier League Model")

    # Load and Explore Data
    file_path = st.text_input("Enter file path for the dataset:", value="C:\\Users\\Rufeeh\\OneDrive\\Desktop\\Africa-Data-School-Curriculum\\players.csv")
    api_url = st.text_input("Enter API URL for live data:", value="https://fantasy.premierleague.com/api/bootstrap-static/")

    if st.button("Fetch Live Data"):
        live_data = fetch_live_data(api_url)
        if live_data is not None:
            explore_data(live_data)

    data = load_data(file_path)

    if data is not None:
        explore_data(data)

        # Train Model
        model = train_model(data)

        if model:
            st.write("### Model Trained Successfully")

            # Make Predictions
            data = make_predictions(model, data)

            # AI-Driven Transfer Suggestions
            suggest_transfers(data)

            # Track Opponents
            track_opponents(data)

    # Placeholder for future functionalities
    st.write("### Future Features")
    st.markdown("- **Live Match Data Updates**")

if __name__ == "__main__":
    main()
