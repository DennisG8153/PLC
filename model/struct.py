import numpy as np
import keras
from keras.src.layers import LSTM, Dense
from keras.src.models import Sequential
import os

model_path = "./my_model.keras"


class ModelWork:

    def __init__(self):
        if os.path.exists(model_path):
            self.model = self.get_model() # get the trained data if exists
    
        else:
            self.model = self.make_new_model() # train the data if not exist


    def make_new_model(self): # creates a simple model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')


    def get_model(self):
        model = keras.models.load_model(model_path)
        return model
        

    def save_model(self):
        # calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
        self.model.save(model_path)
        # needs train and test arrays