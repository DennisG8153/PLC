import streamlit as st
from datetime import date
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define start and end dates
START_DATE = "2015-01-01"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")

# Custom CSS for setting a dark green theme background with bold text
st.markdown("""
    <style>
    .stApp {
        background-color: #d0f0c0; /* Dark green */
    
    }
    h1, h2, h3, h4, h5, h6, label {

    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar menu for navigation
menu = st.sidebar.selectbox("Navigate", ["Home", "Stock Predictor"])

# Home Screen
if menu == "Home":
    st.title("Welcome to the Stock Prediction Application")
    st.write("This app provides stock prediction insights. Use the sidebar to navigate to the Stock Predictor.")
    st.write("Feel free to explore the stock data and predictions in the Stock Predictor section.")

# Stock Predictor Screen
elif menu == "Stock Predictor":
    st.title("Stock Predictor")

    # Expanded stock options
    stock_options = (
        'GOOGL',    # Google
        'MSFT',     # Microsoft
        'NVDA',     # Nvidia
        'TSLA',     # Tesla
        'AAPL',     # Apple
        'AMZN',     # Amazon
        'NKE',      # Nike
        'SPOT',     # Spotify
        'TSM',      # Taiwan Semiconductor
        'NFLX',     # Netflix
        'RBLX'      # Roblox
    )
    chosen_stock = st.selectbox('Choose a stock for prediction:', stock_options)

    # Prediction period in years
    years_to_predict = st.slider('Select number of years to forecast:', 1, 4)
    forecast_days = years_to_predict * 365

    @st.cache_data
    def fetch_data(ticker):
        """Fetches stock data from Yahoo Finance and resets the index."""
        stock_data = yf.download(ticker, START_DATE, CURRENT_DATE)
        stock_data.reset_index(inplace=True)
        return stock_data

    # Display loading message
    loading_message = st.text('Retrieving stock data...')
    stock_data = fetch_data(chosen_stock)
    loading_message.text('Stock data retrieved successfully!')

    # Show raw data
    st.subheader('Recent Stock Data')
    st.write(stock_data.tail())

    # Plotting the historical closing prices
    st.subheader(f'Historical Closing Prices of {chosen_stock}')
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=stock_data['Date'], y=stock_data['Close'].squeeze(), label='Closing Price')
    plt.title(f'Closing Price of {chosen_stock} Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.xticks(rotation=45)
    st.pyplot(plt)
