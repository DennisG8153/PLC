import pandas as pd
import numpy as np
from keras import callbacks
from nn_wrapper import ModelWork # import my own class 


def lstm_prediction_data(data: pd.DataFrame, time_steps: int, days_out: int):
    # cuts into "input data", & "target prediction data"
    ####################################################

    # time_steps = 10
    # e.g. len(data) = 200
    # days out = 14
    input_data, target_prediction_data = [], []
    # range = 10 -> 185 
        # ranges are exclusive to ending index so 186-1 = '185'
    for i in range(time_steps, len(data) - days_out): # starts w every 10 days
        # if we need to predict 14 days in the future and assess it we have to stop it 14 days in the past
        # iloc[10 - 10 = 0: 10]
        input_data.append(data.iloc[(i - time_steps):i,:]) # training data is 10 by 5
        target_prediction_data.append(data.iloc[i + days_out, 3]) # 3 is "close" column
        # iloc[200 - 10 - 1: 200]
    
    return np.array(input_data), np.array(target_prediction_data)


def train_model(model_wrapper: ModelWork, data: pd.DataFrame):
    keras_model = model_wrapper.model # accesses model property (keras model) of ModelWork
    data_used_to_predict, data_to_predict = lstm_prediction_data(data, time_steps = model_wrapper.timestep, days_out = model_wrapper.future)
    # Add early stopping to prevent overfitting
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # Train the model
    keras_model.fit(
        data_used_to_predict,
        data_to_predict,
        epochs = 50,
        batch_size = 32,
        callbacks=[early_stopping]
    )