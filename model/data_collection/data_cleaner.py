import pandas as pd
import os
# import matplotlib.pyplot as plt
# from datetime import date
from sklearn.preprocessing import MinMaxScaler


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


def normalize_data(stock_data: pd.DataFrame):
    # scale the data to be between 0 and 1
    # when scaling it normalizes both test and train data with respect to training data
    ###################################################################################

    scaler = MinMaxScaler()
    stock_data[stock_data.columns] = scaler.fit_transform(stock_data)
    return stock_data, scaler


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
# }
######################################################################
# "AAPL_stock_data.csv" .split("_") --> ["AAPL", "stock", "data.csv"]
# ["AAPL", "stock", "data.csv"][0] --> "AAPL"
######################################################################


def load_all_data(folder_path: str):
    datas_to_return = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            # directory path to read files (it is in a different directory)
            saved_data = read_data_file(os.path.join(dirpath, filename)) 
            train_set, test_set = cut_data(saved_data)

            datas_to_return.append(
                {
                    "ticker": filename.split("_")[0], 
                    "test_set": test_set,
                    "train_set": train_set
                }
            )
    return datas_to_return


if __name__ == "__main__":
    for dirpath, dirnames, filenames in os.walk("../../data/raw"):
        print("Current directory:", dirpath)
        print("Files:", filenames)
        for file in filenames: 
            saved_data = read_data_file("../../data/raw/"+file)
            print(saved_data)
            # write_data_file("../../data/raw/"+file, saved_data)