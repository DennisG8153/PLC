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
    loaded_nn = ModelWork(model_path)

    # ### ----------------------------------------- 3 ----------------------------------------- ###
    # print("checkpoint 3")
    # for loaded_data in all_companies_data: 
    #      train_model(loaded_nn, loaded_data["normalized_train_set"])

    # ### ----------------------------------------- 4 ----------------------------------------- ###
    # print("checkpoint 4")
    # loaded_nn.save_model()

    ### ----------------------------------------- 5 ----------------------------------------- ###
    print("checkpoint 5")
    output_file = f"output_for_{days_to_train_on}_{prediction_start_date}_{days_to_predict}.txt"
    output_file_path = os.path.join(current_directory, "output_files", output_file)
    with open(output_file, "w") as f:
        for loaded_data in all_companies_data:
            company_name = loaded_data["ticker"]
            f.write(f"Company: {company_name}\n")
            f.write("Comparison of Actual vs Predicted Prices:\n")
            f.write(f"{'dates':<12} {'actual':<10} {'predicted':<10}\n")  # Clearer column labels

            dates = pd.to_datetime(loaded_data["normalized_test_set"].index)
            # print(f"Company: {company_name}, Dates: {dates[:5]}...")  # Debug: first 5 dates for inspection

            # Make predictions
            _, predicted_prices = predict_model_test(loaded_nn, loaded_data["normalized_test_set"])
            _, correct_prediction = lstm_prediction_data(loaded_data["raw_test_set"])

            # Debug: check lengths of test set and predictions
            print(f"Company: {company_name}, Test Set Length: {len(dates)}, Predictions Length: {len(correct_prediction)}")

            # Restore prices to their original scale
            restored_prices = loaded_data["normalizer"].restore_price_column(correct_prediction.values)
            restored_predictions = loaded_data["normalizer"].restore_price_column(predicted_prices.values)

            # Debug: print a preview of the restored prices and predictions
            print(f"Company: {company_name}, Restored Prices: {restored_prices[:5]}, Predictions: {restored_predictions[:5]}")

            # Write the comparison table
            for i in range(0, len(restored_prices), 10):
                current_date = dates[i+loaded_nn.prediction_start_date]
                f.write(f"{current_date.strftime('%Y-%m-%d'):<12} {restored_prices[i][0]:<10.2f} {restored_predictions[i][0]:<10.2f}\n")
                

            # Handle the final date
            final_date = dates[len(dates) - 1] + timedelta(days=loaded_nn.prediction_start_date)
            f.write(f"{final_date.strftime('%Y-%m-%d'):<12} {'N/A':<10} {restored_predictions[-1]}\n")

            f.write("\n")  # Separate companies with a newline for better readability