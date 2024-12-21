import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
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
    st.write("### Dataset Overview")
    if data.empty:
        st.write("No data available to display.")
        return

    st.write(data.head())
    st.write("### Data Summary")
    st.write(data.describe())
    st.write("### Missing Values")
    st.write(data.isnull().sum())

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
    st.write("### Training Prediction Model")

    features = [
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bonus', 'influence', 'creativity', 'threat'
    ]
    target = 'total_points'

    if not all(col in data.columns for col in features + [target]):
        st.error("Required columns for training are missing in the dataset.")
        return None

    X = data[features]
    y = data[target]

    X = X.fillna(0)
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter Tuning
    st.write("### Optimizing Model Hyperparameters")
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [500, 700],
        'max_depth': [30, 50, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.4f}")

    return model, features

# Display Feature Importances
def display_feature_importances(model, features):
    st.write("### Feature Importances")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importances")
    st.plotly_chart(fig)

# Make Predictions
def make_predictions(model, data, features):
    st.write("### Make Predictions")

    if not all(col in data.columns for col in features):
        st.error("Required columns for predictions are missing in the dataset.")
        return

    X = data[features]
    X = X.fillna(0)

    predictions = model.predict(X)
    data['predicted_points'] = predictions

    st.write("### Predicted Points")
    st.write(data[['name', 'predicted_points']].sort_values(by='predicted_points', ascending=False).head(10))

    st.write("### Predicted Points Distribution")
    fig = px.histogram(data, x='predicted_points', nbins=20, title='Distribution of Predicted Points')
    st.plotly_chart(fig)

    return data

# Evaluate Team and Predict Haul
def evaluate_team(data):
    st.write("### Team Evaluation and Haul Prediction")

    team = st.multiselect("Select your team players:", options=data['name'].unique())
    haul_threshold = st.slider("Set points threshold for a haul:", min_value=50, max_value=150, value=75)

    if team:
        team_data = data[data['name'].isin(team)]
        total_predicted_points = team_data['predicted_points'].sum()

        st.write("#### Selected Team Players:")
        st.write(team_data[['name', 'predicted_points']])

        st.write(f"#### Total Predicted Points: {total_predicted_points:.2f}")
        if total_predicted_points >= haul_threshold:
            st.success(f"Your team is expected to haul with {total_predicted_points:.2f} predicted points!")
        else:
            st.warning(f"Your team may fall short of a haul, predicted to score {total_predicted_points:.2f} points.")

# AI-Driven Transfer Suggestions
def suggest_transfers(data):
    st.write("### AI-Driven Transfer Suggestions")
    transfer_in_candidates = data[['name', 'predicted_points']].sort_values(by='predicted_points', ascending=False).head(5)
    st.write("#### Top Players to Transfer In")
    st.write(transfer_in_candidates)

    if 'form' in data.columns:
        transfer_out_candidates = data[data['form'] < 3][['name', 'form']].sort_values(by='form').head(5)
        st.write("#### Players to Consider Transferring Out")
        st.write(transfer_out_candidates)
    else:
        st.write("Form data not available. Cannot suggest players to transfer out.")

# Track Opponent Behavior
def track_opponents(data):
    st.write("### Opponent Tracking")

    if 'chip_usage' in data.columns:
        chip_counts = data['chip_usage'].value_counts()
        st.write("#### Chip Usage Summary")
        st.bar_chart(chip_counts)
    else:
        st.write("No chip usage data available.")

    if 'budget' in data.columns:
        st.write("#### Budget Flexibility")
        fig = px.box(data, x='budget', title='Distribution of Remaining Budgets')
        st.plotly_chart(fig)
    else:
        st.write("No budget data available.")

# Main Function
def main():
    st.title("FPL Prediction Model")

    # File Input
    file_path = st.text_input("Enter file path for the dataset:", value="C:\\Users\\Rufeeh\\OneDrive\\Desktop\\Africa-Data-School-Curriculum\\players.csv")
    api_url = st.text_input("Enter API URL for live data:", value="https://fantasy.premierleague.com/api/bootstrap-static/")

    if st.button("Fetch Live Data"):
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
            evaluate_team(data)
            suggest_transfers(data)
            track_opponents(data)

if __name__ == "__main__":
    main()
