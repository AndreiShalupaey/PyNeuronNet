import numpy as np

def sig(x):
    return 1 / (1 + np.exp(-x)) 

def deriv_sig(x):
    return sig(x) * (1 - sig(x))
    