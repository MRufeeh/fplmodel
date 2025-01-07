import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.express as px
import requests

# Load Data
@st.cache_data
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
            return pd.json_normalize(live_data['elements'])
        else:
            st.error(f"Failed to fetch live data. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred while fetching live data: {e}")
        return None

# Data Exploration and Visualization
def explore_data(data):
    st.write("## Data Overview")
    if data.empty:
        st.write("No data available to display.")
        return

    with st.expander("📋 Dataset Sample", expanded=True):
        st.write("### Sample Data")
        st.dataframe(data.head())

    with st.expander("📊 Summary Statistics"):
        st.write("### Data Summary")
        st.dataframe(data.describe())

    with st.expander("🔍 Missing Values"):
        st.write("### Missing Values")
        st.dataframe(data.isnull().sum())

    with st.expander("🌡️ Correlation Heatmap"):
        st.write("### Correlation Heatmap")
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.write("Not enough numeric data for correlation heatmap.")

# Train Prediction Model
def train_model(data):
    st.write("## Training Prediction Model")

    features = [
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bonus', 'influence', 'creativity', 'threat'
    ]
    target = 'total_points'

    if not all(col in data.columns for col in features + [target]):
        st.error("Required columns for training are missing in the dataset.")
        return None, None

    X = data[features].fillna(0)
    y = data[target].fillna(0)

    # Normalize data using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Stacking with optimized hyperparameters
    st.write("### Using Model Stacking")
    base_models = [
        ('ridge', Ridge(alpha=1.0)),
        ('random_forest', RandomForestRegressor(n_estimators=1000, max_depth=25, random_state=42)),
        ('gbr', GradientBoostingRegressor(n_estimators=800, max_depth=5, learning_rate=0.1, random_state=42))
    ]
    stack_model = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression()
    )
    stack_model.fit(X_train, y_train)

    y_pred = stack_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.4f}")

    return stack_model, features

# Display Feature Importances
def display_feature_importances(model, features):
    with st.expander("🔑 Feature Importances", expanded=True):
        st.write("## Feature Importances")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.mean([est.feature_importances_ for est in model.estimators_ if hasattr(est, 'feature_importances_')], axis=0)

        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importances")
        st.plotly_chart(fig)

# Make Predictions
def make_predictions(model, data, features):
    with st.expander("📈 Predictions", expanded=True):
        st.write("## Making Predictions")

        if not all(col in data.columns for col in features):
            st.error("Required columns for predictions are missing in the dataset.")
            return data

        scaler = StandardScaler()
        X = data[features].fillna(0)
        X_scaled = scaler.fit_transform(X)

        predictions = model.predict(X_scaled)
        data['predicted_points'] = predictions

        st.write("### Top Predicted Players")
        st.dataframe(data[['name', 'predicted_points']].sort_values(by='predicted_points', ascending=False).head(10))

        st.write("### Predicted Points Distribution")
        fig = px.histogram(data, x='predicted_points', nbins=20, title='Distribution of Predicted Points')
        st.plotly_chart(fig)

        return data

# Fantasy Team Selection
def fantasy_team_selector(data):
    with st.expander("⚽ Fantasy Team Selector", expanded=True):
        st.write("## Fantasy Team Selector")

        if "name" not in data.columns:
            st.error("Player names are missing from the dataset.")
            return

        # Budget and selection constraints
        budget = 100.0
        max_players = 15

        # Player selection
        selected_players = st.multiselect("Select players for your team:", data['name'])
        selected_data = data[data['name'].isin(selected_players)]

        total_cost = selected_data['now_cost'].sum() / 10
        st.write(f"### Selected Players ({len(selected_players)}/{max_players})")
        st.write(f"**Total Cost:** £{total_cost}m / £{budget}m")

        if total_cost > budget:
            st.error("Your selected team exceeds the budget!")
        elif len(selected_players) > max_players:
            st.error("You can only select up to 15 players!")
        else:
            st.success("Team selection is valid.")

        st.write("### Selected Team")
        st.dataframe(selected_data[['name', 'team', 'position', 'predicted_points', 'now_cost']])

# Main Function
def main():
    st.title("Fantasy Premier League (FPL) Prediction and Team Selector")

    st.sidebar.title("Settings")
    file_path = "C:\\Users\\Rufeeh\\OneDrive\\Desktop\\Africa-Data-School-Curriculum\\players.csv"
    api_url = st.sidebar.text_input("Enter API URL:", "https://fantasy.premierleague.com/api/bootstrap-static/")

    if st.sidebar.button("Fetch Live Data"):
        live_data = fetch_live_data(api_url)
        if live_data is not None:
            explore_data(live_data)

    data = load_data(file_path)

    if data is not None:
        explore_data(data)

        model, features = train_model(data)

        if model:
            display_feature_importances(model, features)
            data = make_predictions(model, data, features)
            fantasy_team_selector(data)

if __name__ == "__main__":
    main()
