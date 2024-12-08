import pandas as pd
from nn_wrapper import ModelWork
from attributes.training import lstm_prediction_data # import my own function 
# input data has to be 10 by 5

def predict_model(model_wrapper: ModelWork, data_to_predict: pd.DataFrame):
    keras_model = model_wrapper.model # sets our keras model to our ModelWork class
    
    # given x (prediction_input_data) data it will predict y 
    prediction_input_data, correct_prediction = lstm_prediction_data(data_to_predict, model_wrapper.timestep)

    #last_prices_scaled = scaler.transform(last_prices.reshape(-1, 1))

    predicted_prices = keras_model.predict(prediction_input_data)
    
    #predicted_prices = scaler.inverse_transform(predicted_prices_scaled)
    df_correct_prediction = pd.DataFrame(correct_prediction)
    df_predicted_prices = pd.DataFrame(predicted_prices)
    
    error = df_correct_prediction.compare(df_predicted_prices)
    print(error, error.std())

    return