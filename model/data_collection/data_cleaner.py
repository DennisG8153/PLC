import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

def create_lstm_data(data, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0]) 
        y.append(data[i + time_steps, 0])
    return np.array(x), np.array(y)