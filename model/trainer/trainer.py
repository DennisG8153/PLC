import keras
from sklearn.preprocessing import MinMaxScaler


def train_model(model: keras.Model):
    model.fit(x, y, epochs = 50, batch_size = 32)