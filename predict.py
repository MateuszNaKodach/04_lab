# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np

AXIS_ROWS = 0
AXIS_COLUMNS = 1


# wczytuje dane ze zbioru

def get_main_data():
    x_data, y_data = pkl.load(open('train.pkl', mode='rb'))
    x_data = x_data[0:6000]
    y_data = y_data[0:6000]
    return x_data, y_data

# wybieram dane uczace, 70 procent zbioru
def get_learn_data():
    data = get_main_data()
    x_train = data[0]
    y_train = data[1]

    #return (x_train)[0:int(x_train.shape[AXIS_ROWS] * 0.7)], y_train[0:int(y_train.shape[AXIS_ROWS] * 0.7)]
    return (x_train)[0:int(x_train.shape[AXIS_ROWS] * 0.7)], y_train[0:int(y_train.shape[AXIS_ROWS] * 0.7)]


def get_validate_data():
    x_validate = get_main_data()[0]
    y_validate = get_main_data()[1]

    return (x_validate)[int(x_validate.shape[AXIS_ROWS] * 0.7):x_validate.shape[AXIS_ROWS]], y_validate[int(
        x_validate.shape[AXIS_ROWS] * 0.7):x_validate.shape[AXIS_ROWS]]


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    #uint16
    return X.astype(np.uint16) @ ~(X_train.transpose().astype(bool)) + ~(X.astype(bool)) @ X_train.astype(np.uint16).transpose()


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    x_train = get_learn_data()[0]
    y_train = get_learn_data()[1]


    x_valid = get_validate_data()[0]
   # y_valid = get_validate_data()[1]

    result = hamming_distance(x_valid,x_train)

    print(result)
    pass
