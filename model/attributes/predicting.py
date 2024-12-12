import pandas as pd
import numpy as np
from nn_wrapper import ModelWork
from attributes.training import lstm_prediction_data # import my own function 


def predict_model_test(model_wrapper: ModelWork, data_to_predict: pd.DataFrame):
    keras_model = model_wrapper.model  # Sets our keras model to our ModelWork class

    # generate prediction input data and the corresponding correct labels
    prediction_input_data, correct_prediction = lstm_prediction_data(data_to_predict, model_wrapper.days_to_train_on, model_wrapper.prediction_start_date, model_wrapper.days_to_predict)

    # extract the last relevant portion of the data and stack it correctly
    additional_data = data_to_predict.iloc[(len(data_to_predict) - model_wrapper.days_to_train_on):len(data_to_predict), :].to_numpy()

    # ensure prediction_input_data and additional_data have the correct dimensions
    prediction_input_data = np.concatenate([prediction_input_data, np.array([additional_data])])

    # reshape to ensure it matches LSTM input shape
    prediction_input_data = prediction_input_data.reshape(-1, model_wrapper.days_to_train_on, data_to_predict.shape[1])

    correct_prediction = pd.DataFrame(correct_prediction)
    
    # perform the prediction
    predicted_prices = keras_model.predict(prediction_input_data)
    # print("predicted_prices", predicted_prices)
    predicted_prices = pd.DataFrame(predicted_prices)

    return correct_prediction, predicted_prices
