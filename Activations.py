import numpy as np

def relu(a):
    return np.maximum(a, 0)
        
def drelu(a):
    for i in range(len(a)):
        if a[i] > 0:
            a[i] = 1
        else:
            a[i] = 0
    return a

def linear(a):
    return a

def dlinear(a):
    return 1

methods = {'relu' : relu, 'drelu' : drelu,
           'linear' : linear, 'dlinear' : dlinear}
