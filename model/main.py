import os
import pandas as pd
os.environ["KERAS_BACKEND"] = "tensorflow" 
from nn_wrapper import ModelWork
from data_collection.data_cleaner import load_all_data_normalized
from attributes.training import train_model
from attributes.predicting import predict_model 

# 1. load in saved data from raw folder
# 2. make/load neural network (construct ModelWork)
# 3. invoke trainer on model and data
# 4. save model
# 5. invoke predictor on model and data

## repeat (3)-(5) with new scaling, layers, and weights however many times
## keep best version model 

if __name__ == "__main__":
    print("checkpoint 0")
    current_file = os.path.realpath("main.py") # main.py
    current_directory = os.path.dirname(current_file) # model

    ### ----------------------------------------- 1 ----------------------------------------- ###
    print("checkpoint 1")
    PLC_path = os.path.dirname(current_directory) # PLC
    raw_data_folder_path = os.path.join(PLC_path, "data","raw")
    all_companies_data = load_all_data_normalized(raw_data_folder_path)

    #print(all_companies_data[0]["train_set"])
    #print(all_companies_data[0]["test_set"])

    ### ----------------------------------------- 2 ----------------------------------------- ###
    print("checkpoint 2")
    model_path = os.path.join(current_directory, "saved_models","model_1.keras")
    loaded_nn = ModelWork(model_path)

    ### ----------------------------------------- 3 ----------------------------------------- ###
    print("checkpoint 3")
    for loaded_data in all_companies_data: 
         train_model(loaded_nn, loaded_data["train_set"])

    # ### ----------------------------------------- 4 ----------------------------------------- ###
    print("checkpoint 4")
    loaded_nn.save_model()

    ### ----------------------------------------- 5 ----------------------------------------- ###
    print("checkpoint 5")
    for loaded_data in all_companies_data:
        correct_prediction, predicted_prices = predict_model(loaded_nn, loaded_data["test_set"])
        
        restored_prices = pd.DataFrame(loaded_data["normalizer"].restore_price_column(correct_prediction.values))
        restored_predictions = pd.DataFrame(loaded_data["normalizer"].restore_price_column(predicted_prices.values))
    

        # compares the original test set and the predicted test set
        # training set is irrelevant in the comparison as it is only what it learned from
        error = restored_prices.compare(restored_predictions)
        print(error, error.std())