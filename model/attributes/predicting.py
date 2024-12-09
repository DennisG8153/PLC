import pandas as pd
import numpy as np
from nn_wrapper import ModelWork
from attributes.training import lstm_prediction_data # import my own function 
# input data has to be 10 by 5

# def predict_model_test(model_wrapper: ModelWork, data_to_predict: pd.DataFrame):
#     keras_model = model_wrapper.model # sets our keras model to our ModelWork class
    
#     # given x (prediction_input_data) data it will predict y 
#     prediction_input_data, correct_prediction = lstm_prediction_data(data_to_predict, model_wrapper.timestep, model_wrapper.future)
#     prediction_input_data = np.append(prediction_input_data, data_to_predict.iloc[(len(data_to_predict) - model_wrapper.timestep - 1):len(data_to_predict),:]) # training data is 10 by 5

#     correct_prediction = pd.DataFrame(correct_prediction)
#     predicted_prices = keras_model.predict(prediction_input_data)
#     predicted_prices = pd.DataFrame(predicted_prices)

#     return correct_prediction, predicted_prices

def predict_model_test(model_wrapper: ModelWork, data_to_predict: pd.DataFrame):
    keras_model = model_wrapper.model  # Sets our keras model to our ModelWork class

    # Generate prediction input data and the corresponding correct labels
    prediction_input_data, correct_prediction = lstm_prediction_data(data_to_predict, model_wrapper.timestep, model_wrapper.future)

    # Extract the last relevant portion of the data and stack it correctly
    additional_data = data_to_predict.iloc[(len(data_to_predict) - model_wrapper.timestep - 1):len(data_to_predict), :].to_numpy()

    # Ensure prediction_input_data and additional_data have the correct dimensions
    prediction_input_data = np.concatenate([prediction_input_data, np.array([additional_data])])

    # Reshape to ensure it matches LSTM input shape
    prediction_input_data = prediction_input_data.reshape(-1, model_wrapper.timestep, data_to_predict.shape[1])

    correct_prediction = pd.DataFrame(correct_prediction)
    
    # Perform the prediction
    predicted_prices = keras_model.predict(prediction_input_data)
    predicted_prices = pd.DataFrame(predicted_prices)

    return correct_prediction, predicted_prices
