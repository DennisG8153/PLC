import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import os
from data_collection.data_cleaner import read_data_file, write_data_file

start_date = "2015-01-01"


def combine_data(old_data: pd.DataFrame, new_data: pd.DataFrame):
    # this literally links the data together
    return pd.concat([old_data, new_data]).drop_duplicates()


def download_new_data(ticker):
    # fetch data
    stock_data = yf.download(ticker, start=start_date, end=datetime.today().strftime('%Y-%m-%d'))

    # Filter relevant columns
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return stock_data


def fetch_and_update_data(ticker: str, filename: str):
    ## Fetches and updates stock data for a single ticker with a 5-second wait interval between queries.
    ## Parameters:
    ## - ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL').
    ## - filename (str): file of data.
    ####################################################################################################

    try:
        new_stock_data = download_new_data(ticker)

        try:
            old_stock_data = read_data_file(filename)
            combined_data = combine_data(old_stock_data, new_stock_data)
            write_data_file(filename, combined_data)
        except FileNotFoundError:
            print(f"No existing data file for {ticker}. Creating new one.")

        print(f"Data for {ticker} updated and saved to {filename}.")
    
    except Exception as e:
        print(f"Error during fetching or saving data for {ticker}: {e}")


if __name__ == "__main__":
    # List of stock tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "KO", "PEP", "XOM", "CVX", "UNH", "V", "MA", "HD", "LMT", "MMM", "CAT", "DIS", "T", "VZ", "IBM", "GE", "HON", "MCD", "SBUX", "WMT", "TGT", "COST", "ORCL", "INTC", "CSCO", "TXN", "QCOM", "ABBV", "MRK", "PFE", "BMY", "ABT", "AMGN", "MDT", "BA", "GD", "RTX", "DE", "UPS", "FDX", "AMT", "PLD", "SPG", "BLK", "GS", "JPM", "BAC", "C", "SCHW", "NKE", "TMO", "DHR", "ISRG", "ZTS", "ROP", "CRM", "NOW", "CARR", "APD", "LIN", "ECL", "ADM", "DOW", "FMC", "NEM", "CL", "CHD", "HSY", "MKC", "KMB", "CLX", "SYY", "KO", "LLY", "EL", "ALGN", "ADP", "PAYX", "CCI", "EQIX", "PSA", "O", "KHC", "MO", "PM", "HSY", "MKC", "ADM", "KO", "NEE", "DUK", "SO", "EXC", "ETR"]  # Add more tickers as needed
    current_file = os.path.realpath()
    # path to go up to PLC then back down to data
    # DO NOT CHANGE FILE STRUCTURE!!!!!!!!!!!
    PLC_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    raw_data_folder_path = os.path.join(PLC_path, "data","raw")

    for ticker in tickers:
        # print statement that states it is fetching data for each ticker 
        print(f"Fetching data for {ticker} from {start_date} to {datetime.today().strftime('%Y-%m-%d')}...")
        filename = os.path.join(raw_data_folder_path, f"{ticker}_stock_data.csv")
        fetch_and_update_data(ticker, filename)
        
        # Wait 5 seconds between each stock query
        print(f"Waiting 5 seconds before the next query...")
        time.sleep(5)

    print("All stocks have been processed.")