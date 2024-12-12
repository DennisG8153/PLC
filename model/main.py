import os
os.environ["KERAS_BACKEND"] = "tensorflow" 


import pandas as pd
from datetime import timedelta
from nn_wrapper import ModelWork
from data_collection.data_cleaner import load_all_data
from attributes.training import lstm_prediction_data, train_model
from attributes.predicting import predict_model_test 

# 1. load in saved data from raw folder
# 2. make/load neural network (construct ModelWork)
# 3. invoke trainer on model and data
# 4. save model
# 5. invoke predictor on model and data

## repeat (3)-(5) with new scaling, layers, and weights however many times
## keep best version model 

days_to_train_on = 10
prediction_start_date = 1
days_to_predict = 14



if __name__ == "__main__":
    print("checkpoint 0")
    current_file = os.path.realpath("main.py") # main.py
    current_directory = os.path.dirname(current_file) # model

    ### ------------------------------------------- 1 ----------------------------------------- ###
    print("checkpoint 1")
    PLC_path = os.path.dirname(current_directory) # PLC
    raw_data_folder_path = os.path.join(PLC_path, "data","raw")

    # {
    #     ticker: str # ticker of the stock data
    #     entire_data: pd.DataFrame # entire set of stock data
    #     normalizer: normalizes data # company specific due to scaling
    # }
    all_companies_data = load_all_data(raw_data_folder_path)

    # normalizes data
    # saved_data = data_scaler.normalize_entire_data(saved_data) 

    ### ------------------------------------------- 2 ----------------------------------------- ###
    print("checkpoint 2")
    model_file_name  = f"model_{days_to_train_on}_{prediction_start_date}_{days_to_predict}.keras"
    model_path = os.path.join(current_directory, "saved_models", model_file_name)
    loaded_nn = ModelWork(model_path, days_to_train_on, prediction_start_date, days_to_predict)

    ### ----------------------------------------- 3 ----------------------------------------- ###
    print("checkpoint 3")
    for loaded_data in all_companies_data: 
         train_model(loaded_nn, loaded_data["normalized_train_set"])

    ### ----------------------------------------- 4 ----------------------------------------- ###
    print("checkpoint 4")
    loaded_nn.save_model()

    ### ----------------------------------------- 5 ----------------------------------------- ###
    print("checkpoint 5")
    output_file = f"output_for_{days_to_train_on}_{prediction_start_date}_{days_to_predict}.txt"
    output_file_path = os.path.join(current_directory, "output_files", output_file)
    with open(output_file_path, "w") as f:
        for loaded_data in all_companies_data:
            company_name = loaded_data["ticker"]
            f.write(f"Company: {company_name}\n")
            f.write("Comparison of Actual vs Predicted Prices:\n")
            f.write(f"{'dates'} {'actual'} {'predicted'}\n")  # clearer column labels

            dates = pd.to_datetime(loaded_data["normalized_test_set"].index)

            # make predictions
            _, predicted_prices = predict_model_test(loaded_nn, loaded_data["normalized_test_set"])
            _, correct_prediction = lstm_prediction_data(loaded_data["raw_test_set"], days_to_train_on, prediction_start_date, days_to_predict)

            # print(predicted_prices)
            # restore prices to their original scale
            restored_predictions = loaded_data["normalizer"].restore_price_column(predicted_prices.values)

            # debug: print a preview of the restored prices and predictions
            print(f"Company: {company_name}, Restored Prices: {correct_prediction[:5]}, Predictions: {restored_predictions[:5]}")

            # write the comparison table
            for i in range(0, len(correct_prediction), 10):
                current_date = dates[i+prediction_start_date + days_to_train_on]
                f.write(f"{current_date.strftime('%Y-%m-%d'):<12} {correct_prediction[i][0]:<10.2f} {restored_predictions[i][0]:<10.2f}\n")

            # handle the final date
            current_date = pd.Timestamp.today().date()
            future_date = current_date + timedelta(days = (4*days_to_predict))
            business_date_range = pd.bdate_range(current_date, future_date)
            for i in range(0,days_to_predict):
                f.write(f"{business_date_range[i].strftime('%Y-%m-%d'):<12} {'N/A':<10} {restored_predictions[-(days_to_predict-i)][0]:<10.2f}\n")
            f.write("\n")  # separate companies with a newline for better readability

            # difference_of_df = correct_prediction - restored_predictions
            # variance = difference_of_df.var()
            # print(difference_of_df)
            # print(correct_prediction)
            # print("CUTOFF")
            # print(restored_predictions)
