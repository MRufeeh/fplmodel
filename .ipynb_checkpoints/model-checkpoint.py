import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import requests


# Fetch FPL API data
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
response = requests.get(url)
fpl_data = response.json()

# Convert player data to DataFrame
players = fpl_data['elements']
fpl_df = pd.DataFrame(players)

kaggle_df = pd.read_csv(r'C:\Users\Rufeeh\OneDrive\Desktop\Africa-Data-School-Curriculum\players.csv')
# Check for missing values in both datasets
print(fpl_df.isnull().sum())
print(kaggle_df.isnull().sum())

















































st.title("FPL MODEL")
