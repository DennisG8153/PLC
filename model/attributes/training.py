import pandas as pd
import numpy as np
from keras import callbacks
from nn_wrapper import ModelWork # import my own class 


def lstm_prediction_data(data: pd.DataFrame, time_steps: int):
    # cuts into "input data", & "target prediction data"
    ####################################################

    input_data, target_prediction_data = [], []
    for i in range(len(data) - time_steps):
        input_data.append(data.iloc[i:(i + time_steps),:]) # training data is 10 by 5
        target_prediction_data.append(data.iloc[i + time_steps, 3]) # 3 is "close" column
    return np.array(input_data), np.array(target_prediction_data)


def train_model(model_wrapper: ModelWork, data: pd.DataFrame):
    keras_model = model_wrapper.model # accesses model property (keras model) of ModelWork
    data_used_to_predict, data_to_predict = lstm_prediction_data(data, time_steps = model_wrapper.timestep)
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