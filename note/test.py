import streamlit as st
from datetime import date
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define start and end dates
start_date = "2015-01-01"
current_date = date.today().strftime("%Y-%m-%d")


st.markdown("""
    <style>
    .stApp {
        background-color: #d0f0c0; /* Dark green */
    
    }
    h1, h2, h3, h4, h5, h6, label {

    }
    </style>
    """, unsafe_allow_html=True)


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
        'GOOGL',   
        'MSFT',    
        'NVDA',     
        'TSLA',     
        'AAPL',     
        'AMZN',    
        'NKE',      
        'SPOT',     
        'TSM',      
        'NFLX',     
        'RBLX'     
    )
    chosestock = st.selectbox('Choose a stock for prediction:', stock_options)

    
    years_to_predict = st.slider('Select number of years to forecast:', 1, 4)
    forecast_days = years_to_predict * 365

    @st.cache_data
    def fetch_data(ticker):
        data = yf.download(ticker, start_date, current_date)
        data.reset_index(inplace=True)
        return data

    # Display loading message
    message = st.text('Retrieving stock data...')
    data = fetch_data(chosestock)
    message.text('Stock data retrieved successfully!')

    # Show raw data
    st.subheader('Recent Stock Data')
    st.write(data.tail())

    # Plotting the historical closing prices
    st.subheader(f'Historical Closing Prices of {chosestock}')
    plt.figure(figsize=(12, 6))
    sns.lineplot(x= data['Date'], y=data['Close'].squeeze(), label='Closing Price')
    plt.title(f'Closing Price of {chosestock} Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.xticks(rotation=45)
    st.pyplot(plt)