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
    with open('train.pkl', 'rb') as f:
        return pkl.load(f)


# wybieram dane uczace, 70 procent zbioru
def get_learn_data():
    x_train = get_main_data()[0]
    y_train = get_main_data()[1]

    return (x_train)[0:int(x_train.shape[AXIS_ROWS] * 0.7)], y_train[0:int(y_train.shape[AXIS_ROWS] * 0.7)]


def get_validate_data():
    x_validate = get_main_data()[0]
    y_validate = get_main_data()[1]

    return (x_validate)[int(x_validate.shape[AXIS_ROWS] * 0.7):x_validate.shape[AXIS_ROWS]], y_validate[int(
        x_validate.shape[AXIS_ROWS] * 0.7):x_validate.shape[AXIS_ROWS]]


def naive_bayess_predictor(x_train, y_train):
    clf = GaussianNB()
    clf.fit(x_train, y_train)  # wylicza parametry do bayessa
    return clf


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    x_train = get_learn_data()[0]
    y_train = get_learn_data()[1]

    #x_valid = get_validate_data()[0]
    #y_valid = get_validate_data()[1]

    nbp = naive_bayess_predictor(x_train, y_train)

    result = np.array(nbp.predict(x)).transpose()
    #result = np.array(nbp.predict(x_valid)).transpose()
    #print((result.transpose()==y_valid).sum())
    #print(np.count_nonzero(result.transpose()==y_valid))
    #print(y_train.shape[AXIS_ROWS])


    return result
