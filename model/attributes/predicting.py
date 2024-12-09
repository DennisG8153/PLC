import pandas as pd
import numpy as np
from nn_wrapper import ModelWork
from attributes.training import lstm_prediction_data # import my own function 
# input data has to be 10 by 5

def predict_model_test(model_wrapper: ModelWork, data_to_predict: pd.DataFrame):
    keras_model = model_wrapper.model # sets our keras model to our ModelWork class
    
    # given x (prediction_input_data) data it will predict y 
    prediction_input_data, correct_prediction = lstm_prediction_data(data_to_predict, model_wrapper.timestep, model_wrapper.future)
    prediction_input_data = np.append(prediction_input_data, data_to_predict.iloc[(len(data_to_predict) - model_wrapper.timestep - 1):len(data_to_predict),:]) # training data is 10 by 5

    correct_prediction = pd.DataFrame(correct_prediction)
    predicted_prices = keras_model.predict(prediction_input_data)
    predicted_prices = pd.DataFrame(predicted_prices)

    return correct_prediction, predicted_prices
