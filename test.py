
import pickle as pkl
import unittest
from predict import predict
from unittest import makeSuite


class TestRunner(unittest.TextTestRunner):
    def __init__(self, result=None):
        super(TestRunner, self).__init__(verbosity=2)

    def run(self):
        suite = TestSuite()
        return super(TestRunner, self).run(suite)

class TestSuite(unittest.TestSuite):
    def __init__(self):
        super(TestSuite, self).__init__()
        self.addTest(makeSuite(TestSigmoid))



class TestSigmoid(unittest.TestCase):
    def test_sigmoid(self):
        data = load_main_data()
        predict(load_main_data())
        self.assertEqual(predict_100_percent(data).all(),data[1].all())


def load_main_data():
    with open('train.pkl', 'rb') as f:
        return pkl.load(f)

def predict_100_percent(x):
    
    return x[1]