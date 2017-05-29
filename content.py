

import pickle as pkl
import numpy as np

AXIS_ROWS = 0
AXIS_COLUMNS = 1
CLASSES_AMOUNT = 36
NN_K = 3

# wczytuje dane ze zbioru - biore 6 tys egzemplarzy
def get_main_data():
    x_data, y_data = pkl.load(open('train.pkl', mode='rb'))
    #x_data = x_data[0:6000]
    #y_data = y_data[0:6000]
    x_data = x_data[0:9000]
    y_data = y_data[0:9000]
    return x_data, y_data


# wybieram dane uczace, 70 procent zbioru
def get_learn_data():
    data = get_main_data()
    x_train = data[0]
    y_train = data[1]

    return (x_train)[0:int(x_train.shape[AXIS_ROWS] * 0.7)], y_train[0:int(y_train.shape[AXIS_ROWS] * 0.7)]


#wybieram dane validacyjne, 30 procent zbioru
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

def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    return y[Dist.argsort(kind='mergesort')]

def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """

    columnCount = y.shape[AXIS_COLUMNS]
    resizedArray = np.delete(y, range(k, columnCount), axis=AXIS_COLUMNS)
    countedClassesOccurences = np.apply_along_axis(np.bincount, axis=1, arr=resizedArray, minlength=CLASSES_AMOUNT + 1)
    countedClassesOccurences = np.delete(countedClassesOccurences, 0, axis=1)
    #bez dzielenia, bo i tak liczy się licznik
    #probabilityOfEachClass = np.divide(countedClassesOccurences, k)

    return countedClassesOccurences

def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """

    p_y_x = np.fliplr(p_y_x)
    y_truea = p_y_x.shape[1] - np.argmax(p_y_x, axis=1)
    y_truea = np.subtract(y_truea, y_true)
    diff = np.count_nonzero(y_truea)
    diff /= y_true.shape[0]

    return diff

def run_program():
    validate_data = get_validate_data()
    x_valid = validate_data[0]
    y_valid = validate_data[1]
    from predict import predict
    predicted = predict(x_valid)
    print("BLAD KLASYFIKACJI:")
    print(classification_error(predicted,y_valid))
    """
    print("PREDICTED")
    print(predicted)
    print("VALID")
    print(y_valid)
    print()
    """
