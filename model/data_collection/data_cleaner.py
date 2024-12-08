import pandas as pd
import os
# import matplotlib.pyplot as plt
# from datetime import date
from sklearn.preprocessing import MinMaxScaler

class Normalizer:
    def __init__(self): # constructor
        self.price_scaler = MinMaxScaler() # scales price (float) columns 
        self.volume_scaler = MinMaxScaler() # scales volume (int) column


    ###################################################################################
    def train_price_column(self, stock_data_column: pd.DataFrame):
        ## trains on 1 price column
        ## DOES NOT TRANSFORM
        self.price_scaler.partial_fit(stock_data_column) 
        ## partial fits because everytime you call "fit()" 
        ## it overwrites previously remembered mins & maxes
        ## BUT in partial fit it remembers all previous data passed in
        ## its the difference between knowing the min & max of only the last column passed in (fit)
        ## vs 
        ## the difference between the min & max of all the columns (partial fit)
        return
    
    def normalize_price_column(self, stock_data_column: pd.DataFrame): 
        # transforms 1 price column 
        stock_data_column = self.price_scaler.transform(stock_data_column)    
        return stock_data_column
    
    def restore_price_column(self, stock_data_column: pd.DataFrame):
        # un-transforms 1 price column 
        stock_data_column = self.price_scaler.inverse_transform(stock_data_column)  
        return stock_data_column
    
    
    ###################################################################################
    def train_and_normalize_volume(self, stock_data: pd.DataFrame):
        # transforms our singular volume column
        # we do not need to break up fit & transform because volume is already a single column
        stock_data[["Volume"]] = self.volume_scaler.fit_transform(stock_data[["Volume"]])
        return stock_data
    
    def restore_volume(self, stock_data: pd.DataFrame):
        # un-transform volume column
        stock_data[["Volume"]] = self.volume_scaler.inverse_transform(stock_data[["Volume"]])
        return stock_data
    

    ###################################################################################
    ###################################################################################
    def normalize_entire_data(self, stock_data: pd.DataFrame):
        # actually calling above functions
        # loop through all columns and train on each on of them 1 at a time
        # this ensures only one column is transformed at once
        for col_name in ["Open","High","Low","Close"]:
            stock_data[[col_name]] = self.train_price_column(stock_data[[col_name]].values)

        for col_name in ["Open","High","Low","Close"]:
            stock_data[[col_name]] = self.normalize_price_column(stock_data[[col_name]].values)
        
        # volume is only one column so we do not have to loop
        stock_data = self.train_and_normalize_volume(stock_data)
        return stock_data
    
    def restore_entire_data(self, stock_data: pd.DataFrame):
        # loops through all columns and restores 1 at a time
        for col_name in ["Open","High","Low","Close"]:
            stock_data[[col_name]] = self.restore_price_column(stock_data[[col_name]].values)

        # restores the volume (only one column no loop)
        stock_data = self.restore_volume(stock_data)
        return stock_data
    

    ###################################################################################
    def clear_scale(self):
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()



def read_data_file(filename: str):
    # reads the data
    data_read_from_file = pd.read_csv(
        filename,
        index_col=0,
        parse_dates=True,
        date_format='%Y-%m-%d'
    )
    return data_read_from_file


def write_data_file(filename: str, old_data_to_write: pd.DataFrame):
    # save updated data to the file in the specified folder
    old_data_to_write.to_csv(filename)


def cut_data(data: pd.DataFrame):
    training_size = round(len(data) * 0.80)
    train_data = data[:training_size]
    test_data  = data[training_size:]
    return train_data, test_data 


######################################################################
# {
#     ticker: str # ticker of the stock data
#     test_set: pd.DataFrame # testing set of stock data
#     train_set: pd.DataFrame # training set of stock data
#     normalizer: normalizes data # company specific due to scaling
# }
######################################################################
# "AAPL_stock_data.csv" .split("_") --> ["AAPL", "stock", "data.csv"]
# ["AAPL", "stock", "data.csv"][0] --> "AAPL"
######################################################################


from typing import TypedDict

class ReturnCompanyData(TypedDict):
    ticker: str
    test_set: pd.DataFrame
    train_set: pd.DataFrame
    normalizer: Normalizer


######################################################################


def load_all_data_normalized(folder_path: str) -> list[ReturnCompanyData]:
    datas_to_return = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            data_scaler = Normalizer()
            
            # reads the data
            saved_data = read_data_file(os.path.join(dirpath, filename)) 
            # normalizes data
            saved_data = data_scaler.normalize_entire_data(saved_data) 
            # cuts our data in 2 (80:20)
            train_set, test_set = cut_data(saved_data)

            # makes returned object
            # dictionary made for each company
            returned_company_data: ReturnCompanyData = {
                "ticker": filename.split("_")[0], 
                "test_set": test_set,
                "train_set": train_set,
                "normalizer": data_scaler
            }

            # adds the dictionaries of company datas to big array
            datas_to_return.append(returned_company_data)

    return datas_to_return


if __name__ == "__main__":
    for dirpath, dirnames, filenames in os.walk("../../data/raw"):
        print("Current directory:", dirpath)
        print("Files:", filenames)
        for file in filenames: 
            saved_data = read_data_file("../../data/raw/"+file)
            print(saved_data)
            # write_data_file("../../data/raw/"+file, saved_data)