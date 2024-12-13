import streamlit as st
from datetime import date
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import SentimentGraphs
import numpy as np

# Define start and end dates
start_date = "2020-01-01"
current_date = date.today().strftime("%Y-%m-%d")

stock_choices = [
    'AAPL', 'ABBV', 'ABT', 'ADM', 'ADP', 'ALGN', 'AMGN', 'AMT', 'AMZN', 'APD',
    'BA', 'BAC', 'BLK', 'BMY', 'C', 'CARR', 'CAT', 'CCI', 'CHD', 'CL',
    'CLX', 'COST', 'CRM', 'CSCO', 'CVX', 'DE', 'DHR', 'DIS', 'DOW', 'DUK',
    'ECL', 'EL', 'EQIX', 'ETR', 'EXC', 'FDX', 'FMC', 'GD', 'GE', 'GOOGL',
    'GS', 'HD', 'HON', 'HSY', 'IBM', 'INTC', 'ISRG', 'JNJ', 'JPM', 'KHC',
    'KMB', 'KO', 'LIN', 'LLY', 'LMT', 'MA', 'MCD', 'MDT', 'MKC', 'MMM',
    'MO', 'MRK', 'MSFT', 'NEE', 'NEM', 'NKE', 'NOW', 'O', 'ORCL', 'PAYX',
    'PEP', 'PFE', 'PG', 'PLD', 'PM', 'PSA', 'QCOM', 'ROP', 'RTX', 'SBUX',
    'SCHW', 'SO', 'SPG', 'SYY', 'T', 'TGT', 'TMO', 'TXN', 'UNH', 'UPS',
    'V', 'VZ', 'WMT', 'XOM', 'ZTS'
]

# Parse the data
def parse_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read()

        companies = data.split("Company:")[1:]

        parsed_data = {}

        for company_block in companies:
            lines = company_block.strip().split('\n')
            company_name = lines[0].strip()
            rows = [line.split() for line in lines[2:]]
            df = pd.DataFrame(rows, columns=['dates', 'historical', 'predicted'])

            df['historical'] = pd.to_numeric(df['historical'], errors='coerce')
            df['predicted'] = pd.to_numeric(df['predicted'], errors='coerce')
            df['dates'] = pd.to_datetime(df['dates'], errors='coerce')

            parsed_data[company_name] = df

        return parsed_data
    except FileNotFoundError:
        st.error(f"The file '{file_path}' wasn't found. Please check the file path and try again.")
        return {}
    except Exception as e:
        st.error(f"Oops! Something went wrong while processing the file: {e}")
        return {}

# Load data
file_path = "./predictedprices.txt"
companyData = parse_data(file_path)

#CSS stuff for styling of website
st.markdown("""
    <style>
    .stApp {
        background-color:  #90EE90; 
    }

    div[data-baseweb="select"] input {
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

## navigation
menu = st.sidebar.selectbox("Navigate", ["Home", "Stock Predictor", "Sentiment Analysis"])

if menu == "Home":
    st.title("Welcome to Moneywise: :wave: :dollar:")
    st.write("This app provides stock prediction insights. Use the sidebar to navigate to the Stock Predictor.")
    st.write("We also feature a sentiment analysis portion for more stock analysis")
    st.image("GOTHAM KNIGHTS.jpg", width=300)

elif menu == "Stock Predictor":
    st.title("Stock Predictor and General Overview")

    count = st_autorefresh(interval=15000, key="refresh")
 
    # allow user to seleect stock
    chosestock = st.selectbox('Choose a stock please :smile:', stock_choices)

    if chosestock:
        @st.cache_data
        def fetch_data(ticker):
            data = yf.download(ticker, start_date, current_date)
            data.reset_index(inplace=True)
            data['Date'] = data['Date'].dt.date 
            return data

        #trying to get live prices refresh as well
        def livePrice(ticker):
            stock = yf.Ticker(ticker)
            price = stock.history(period="1d")['Close'].iloc[-1]
            return price

        data = fetch_data(chosestock)

        livep = livePrice(chosestock)

        if livep is not None:
            st.metric(label=f"Live Price of {chosestock}", value=f"${livep:.2f}")

        st.subheader('Recent Stock Data')
        st.write(data.tail())

        # closing price
        st.subheader(f'Closing Prices of {chosestock}')
        plt.figure(figsize=(11, 7))
        plt.plot(data['Date'], data['Close'].squeeze(), color='blue', label='Closing Price')
        plt.title(f'Closing Price of {chosestock} Over Time', size='x-large', color='red')
        plt.xlabel('Date')
        plt.ylabel('USD')
        plt.xticks(rotation=45)
        st.pyplot(plt)


        data['Daily Return'] = data['Adj Close'].pct_change()
        volatility = data['Daily Return'].std() * np.sqrt(252)

        st.subheader(f"Volatility of the following stock: {chosestock}")
        st.write(f"Annualized Volatility: {volatility * 100:.2f}%")
    

        #calculating volatility
        v = data['Daily Return'].rolling(window=21).std() * (np.sqrt(252))
        plt.figure(figsize=(11, 7))
        plt.plot(data['Date'], v, color='red')
        plt.title(f'Rolling Volatility of {chosestock}',color = 'blue')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.xticks(rotation=45)
        st.pyplot(plt)

       #graphing things from the LSTM predicted prices onto our graphs
        if chosestock in companyData:
            st.subheader(f"Actual vs Predicted Prices for {chosestock}")
            storedStock = companyData[chosestock]

            plt.figure(figsize=(11, 7))
            plt.plot(storedStock['dates'],storedStock['historical'], label="Actual Price", marker='o')
            plt.plot(storedStock['dates'], storedStock['predicted'], label="Predicted Price", linestyle='--', color='red')
            plt.title(f"{chosestock} - Actual vs Predicted Prices",color='blue')
            plt.xlabel("Date")
            plt.ylabel("USD")
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)


    
elif menu == "Sentiment Analysis":
    
    SentimentGraphs.draw()




























