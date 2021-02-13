import numpy as np

from NeuronNet import *

net = NeuronNet()

learn_inputs = np.array([[1, 0], [0, 0], [0, 1]])
learn_answers = np.array([1, 0, 0])

net.learn(3, learn_inputs, learn_answers)

x = np.array([0, 1])

print(net.activate(x))
