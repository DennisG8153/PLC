#from tensorflow import keras 
import keras
from keras import layers, models
import os


class ModelWork:

    def __init__(self, model_path):
        self.timestep = 10
        self.future = 14
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
        # you take 9 days and predict the 10th (10-1=9)
        neural_network.add(layers.LSTM(units = 50, return_sequences = True, input_shape = (self.timestep, 5))) 
        # neural network = [LSTM(10x5)->]
        neural_network.add(layers.LSTM(units = 50))
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