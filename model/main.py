from datetime import timedelta
import os
import pandas as pd
os.environ["KERAS_BACKEND"] = "tensorflow" 
from nn_wrapper import ModelWork
from data_collection.data_cleaner import load_all_data_normalized
from attributes.training import train_model
from attributes.predicting import predict_model_test 

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

    ### ----------------------------------------- 2 ----------------------------------------- ###
    print("checkpoint 2")
    model_path = os.path.join(current_directory, "saved_models","model_1.keras")
    loaded_nn = ModelWork(model_path)

    ### ----------------------------------------- 3 ----------------------------------------- ###
    print("checkpoint 3")
    for loaded_data in all_companies_data: 
         train_model(loaded_nn, loaded_data["train_set"])

    ### ----------------------------------------- 4 ----------------------------------------- ###
    print("checkpoint 4")
    loaded_nn.save_model()

    ### ----------------------------------------- 5 ----------------------------------------- ###
    print("checkpoint 5")

    output_file = "error_output_with_companies_and_dates.txt"
    with open(output_file, "w") as f:
        for loaded_data in all_companies_data:
            company_name = loaded_data["ticker"]
            f.write(f"Company: {company_name}\n")
            f.write("Comparison of Actual vs Predicted Prices:\n")
            f.write(f"{'dates':<12} {'self':<10} {'other':<10}\n")  # header for the comparison table

            dates = pd.to_datetime(loaded_data["test_set"].index)
            print(f"Company: {company_name}, Dates: {dates[:5]}...")  # print the first 5 dates for inspection

            # make predictions
            correct_prediction, predicted_prices = predict_model_test(loaded_nn, loaded_data["test_set"])

            # print lengths of actual data and predicted data for debugging
            print(f"Company: {company_name}, Test Set Length: {len(dates)}, Predictions Length: {len(correct_prediction)}")

            # normalizes
            restored_prices = loaded_data["normalizer"].restore_price_column(correct_prediction.values)
            restored_predictions = loaded_data["normalizer"].restore_price_column(predicted_prices.values)

            # print the full comparison data for debugging (if needed)
            print(f"Company: {company_name}, Full Comparison Data:")

            # write the comparison table to the file, printing every 10th day based on date intervals
            for i in range(0, len(restored_prices), 10):
                f.write(f"{dates[i].strftime('%Y-%m-%d'):<12} {restored_prices[i][0]:<10.2f} {restored_predictions[i][0]:<10.2f}\n")
            f.write(f"{(dates[len(dates)-1] + timedelta(days = loaded_nn.future)).strftime('%Y-%m-%d'):<12} {restored_predictions[-1][0]:<10.2f}\n")

            f.write("\n")  # Separate companies with a newline for better readability