import numpy as np

from Math import *

class Neuron:
    def __init__(self, number_of_weights = 1):

        self.w = np.random.normal(size=number_of_weights)
        self.b = np.random.normal()
        
    def activate(self, inputs):
        
        x = np.dot(self.w, inputs) + self.b

        self.x = x
        
        return sig(x)