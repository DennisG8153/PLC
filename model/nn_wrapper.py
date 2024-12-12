#from tensorflow import keras 
import keras
from keras import layers, models
import os


class ModelWork:

    def __init__(self, model_path, days_to_train_on, prediction_start_date, days_to_predict):
        self.days_to_train_on = days_to_train_on
        self.prediction_start_date = prediction_start_date
        self.days_to_predict = days_to_predict
        self.file_path = model_path
        if os.path.exists(model_path):
            self.model = self.get_model() # self.model is type: "keras model" 
        else:
            self.model = self.make_new_model() # self.model is type: "keras model" 


    def make_new_model(self): 
    # simple starting model
    # builds the lstm model
    ########################

        neural_network = keras.Sequential()
        # neural network = []
        neural_network.add(layers.LSTM(units = 50, return_sequences = True, input_shape = (self.days_to_train_on, 5))) 
        # neural network = [LSTM(10x5)->]
        neural_network.add(layers.LSTM(units = 50, return_sequences = True))
        # neural network = [LSTM(10x5)->LSTM()->]
        neural_network.add(layers.Dense(units = 1))
        # neural network = [LSTM(10x5)->LSTM()->Dense()]
        neural_network.compile(optimizer = 'adam', loss = 'mean_squared_error')
        # neural network = [LSTM(10x5)->LSTM()->Dense()->output]
        neural_network.summary()
        return neural_network


    def get_model(self):
        model = models.load_model(self.file_path) # this makes model type: keras model
        return model 
        

    def save_model(self):
        # calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
        self.model.save(self.file_path)
        # needs train and test arrays