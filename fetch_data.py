import yfinance as yf # type: ignore
import pandas as pd # type: ignore
from datetime import datetime
import time
import os

def fetch_and_update_data(ticker, start_date, folder):
    """
    Fetches and updates stock data for a single ticker with a 5-second wait interval between queries.
    
    Parameters:
    - ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL').
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - folder (str): Folder where the CSV files will be saved.
    """
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = os.path.join(folder, f"{ticker}_stock_data.csv")
    try:
        print(f"Fetching data for {ticker} from {start_date} to {datetime.today().strftime('%Y-%m-%d')}...")

        # Fetch stock data
        stock_data = yf.download(ticker, start=start_date, end=datetime.today().strftime('%Y-%m-%d'))

        # Filter relevant columns
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Load existing data if the file exists
        try:
            # Explicitly specify date format and treat first row as header
            existing_data = pd.read_csv(filename, index_col=0, parse_dates=True, date_format='%Y-%m-%d')
            existing_data.index = pd.to_datetime(existing_data.index)  # Ensure the index is datetime type
            stock_data = pd.concat([existing_data, stock_data]).drop_duplicates()
        except FileNotFoundError:
            print(f"No existing data file for {ticker}. Creating new one.")

        # Save updated data to the file in the specified folder
        stock_data.to_csv(filename)
        print(f"Data for {ticker} updated and saved to {filename}.")
    
    except Exception as e:
        print(f"Error during fetching or saving data for {ticker}: {e}")

def update_stocks(tickers, start_date, folder):
    """
    Updates stock data for a list of tickers, from last to first, with a 5-second wait interval between requests.
    
    Parameters:
    - tickers (list): List of stock ticker symbols (e.g., ['AAPL', 'GOOGL']).
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - folder (str): Folder where the CSV files will be saved.
    """
    for ticker in tickers:
        print(f"Starting data fetching for {ticker}...")
        fetch_and_update_data(ticker, start_date, folder)
        
        # Wait 5 seconds between each stock query
        print(f"Waiting 5 seconds before the next query...")
        time.sleep(5)
    
    print("All stocks have been processed.")

if __name__ == "__main__":
    # List of stock tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "KO", "PEP", "XOM", "CVX", "UNH", "V", "MA", "HD", "LMT", "MMM", "CAT", "DIS", "T", "VZ", "IBM", "GE", "HON", "MCD", "SBUX", "WMT", "TGT", "COST", "ORCL", "INTC", "CSCO", "TXN", "QCOM", "ABBV", "MRK", "PFE", "BMY", "ABT", "AMGN", "MDT", "BA", "GD", "RTX", "DE", "UPS", "FDX", "AMT", "PLD", "SPG", "BLK", "GS", "JPM", "BAC", "C", "SCHW", "NKE", "TMO", "DHR", "ISRG", "ZTS", "ROP", "CRM", "NOW", "CARR", "APD", "LIN", "ECL", "ADM", "DOW", "FMC", "NEM", "CL", "CHD", "HSY", "MKC", "KMB", "CLX", "SYY", "KO", "LLY", "EL", "ALGN", "ADP", "PAYX", "CCI", "EQIX", "PSA", "O", "KHC", "MO", "PM", "HSY", "MKC", "ADM", "KO", "NEE", "DUK", "SO", "EXC", "ETR"]  # Add more tickers as needed
    start_date = "2015-01-01"
    folder = "StockInfo"  # Folder where CSV files will be saved

    # Update stocks with 5-second interval between each query and save in the folder
    update_stocks(tickers, start_date, folder)
