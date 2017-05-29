
import unittest
from unittest import makeSuite
from content import run_program


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
        predicted = run_program()
        self.assertEqual(0,0)

