

import pickle as pkl
import numpy as np

# wczytuje dane ze zbioru
def get_main_data():
    x_data, y_data = pkl.load(open('train.pkl', mode='rb'))
    x_data = x_data[0:6000]
    y_data = y_data[0:6000]
    return x_data, y_data
