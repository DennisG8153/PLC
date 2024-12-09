import pandas as pd
import numpy as np
from keras import callbacks
from nn_wrapper import ModelWork # import my own class 


def lstm_prediction_data(data: pd.DataFrame, days_to_train_on, prediction_start_date, days_to_predict):
    # cuts into "input data", & "target prediction data"
    ####################################################

    input_data, target_prediction_data = [], []

    for current_day in range(days_to_train_on, len(data) - (prediction_start_date + days_to_predict)): 
        input_data.append(data.iloc[(current_day - days_to_train_on): current_day, : ]) # ###

        target_prediction_data.append(data.iloc[(current_day + prediction_start_date): current_day + prediction_start_date + days_to_predict, 3]) # 3 is "close" column
    return np.array(input_data), np.array(target_prediction_data)


def train_model(model_wrapper: ModelWork, data: pd.DataFrame):
    keras_model = model_wrapper.model # accesses model property (keras model) of ModelWork
    data_used_to_predict, data_to_predict = lstm_prediction_data(data, model_wrapper.days_to_train_on, model_wrapper.prediction_start_date, model_wrapper.days_to_predict)
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