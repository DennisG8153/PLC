import streamlit as st

from datetime import date
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import ScatterPlot #imports the ScatterPlot script

# Define start and end dates
start_date = "2015-01-01"
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



def parse_data(file_path):
    try:
        # Attempt to open and read the file
        with open(file_path, 'r') as file:
            data = file.read()

        # Split the data by company block and remove the first empty entry
        companies = data.split("Company:")[1:]

        parsed_data = {}

        for company_block in companies:
       
            lines = company_block.strip().split('\n')
            
           
            company_name = lines[0].strip()
            
         
            header = lines[1]  
            
            
            rows = [line.split() for line in lines[2:]]

            # Create a DataFrame from the rows with specified column names
            df = pd.DataFrame(rows, columns=['dates', 'historical', 'predicted'])

            # Convert the 'historical' and 'predicted' columns to numeric, handling errors gracefully
            df['historical'] = pd.to_numeric(df['historical'], errors='coerce')
            df['predicted'] = pd.to_numeric(df['predicted'], errors='coerce')

            # Convert the 'dates' column to datetime format, handling any errors
            df['dates'] = pd.to_datetime(df['dates'], errors='coerce')


            parsed_data[company_name] = df

        return parsed_data

    except FileNotFoundError:
        st.error(f"The file '{file_path}' wasn't found. Please check the file path and try again.")
        return {}
    except Exception as e:
        st.error(f"Oops! Something went wrong while processing the file: {e}")
        return {}




file_path = ".\predictedprices.txt"


data_dict = parse_data(file_path)


# Apply styling
st.markdown("""
    <style>
    .stApp {
        background-color: #d0f0c0; /* Dark green */
    }
    h1, h2, h3, h4, h5, h6, label {
        color: #2b2b2b; /* Dark text */ 
    }
    </style>
    """, unsafe_allow_html=True)


# Sidebar Navigation
menu = st.sidebar.selectbox("Navigate", ["Home", "Stock Predictor", "Sentiment Analysis"])

# Home Screen
if menu == "Home":
    st.title("Welcome to the Stock Prediction Application")
    st.write("This app provides stock prediction insights. Use the sidebar to navigate to the Stock Predictor.")
    st.write("Feel free to explore the stock data and predictions in the Stock Predictor section.")

# Stock Predictor Screen
elif menu == "Stock Predictor":
    st.title("Stock Predictor")

    # Allow the user to type the stock symbol
    chosestock = st.selectbox('Choose a stock please', stock_choices)  # Default value: 'GOOGL'

    if chosestock:
        # Historical Price Data from Yahoo Finance
        @st.cache_data
        def fetch_data(ticker):
            data = yf.download(ticker, start_date, current_date)
            data.reset_index(inplace=True)
            return data

        # Display loading message
        message = st.text('Getting stock data')
        data = fetch_data(chosestock)
        message.text('Stock data retrieved successfully!')

        # Show raw data
        st.subheader('Recent Stock Data')
        st.write(data.tail())

        # Plotting the historical closing prices
        st.subheader(f'Historical Closing Prices of {chosestock}')
        plt.figure(figsize=(11, 7))
        sns.lineplot(x=data['Date'], y=data['Close'].squeeze(), label='Closing Price')
        plt.title(f'Closing Price of {chosestock} Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price in USD')
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Actual vs Predicted Prices
        if chosestock in data_dict:
            st.subheader(f"Actual vs Predicted Prices for {chosestock}")
            df_selected = data_dict[chosestock]

            
            plt.figure(figsize=(11, 7))
            plt.plot(df_selected['dates'], df_selected['historical'], label="Actual Price", marker='o')
            plt.plot(df_selected['dates'], df_selected['predicted'], label="Predicted Price", linestyle='--')
            plt.title(f"{chosestock} - Actual vs Predicted Prices")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.warning("Actual vs Predicted data not found for this stock.")
elif menu == "Sentiment Analysis":
    ScatterPlot.draw()