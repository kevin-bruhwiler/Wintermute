import numpy as np
from Activations import methods

class DenseNeuralNetwork:
    
    def __init__(self, xs, ys, activation, dactivation):
        self.x = np.zeros(xs)
        self.xs = self.x.shape[0]
        self.y = np.zeros(ys)
        self.ys = self.y.shape[0]
        self.w = np.random.rand(ys, xs)
        self.m = np.zeros_like(self.w)
        self.v = np.zeros_like(self.w)
        self.t = 0
        self.activation = methods[activation]
        self.dactivation = methods[dactivation]
    
    def getWeights(self):
        return self.w
        
    def forwardProp(self):
        a = np.dot(self.w, self.x.T)      
        return self.activation(a).T
    
    def loss(self):
        return np.sum(self.forwardProp() - self.y)

    def cost(self):
        return self.forwardProp() - self.y
        
    def trainNetwork(self, z):
        dy = self.dactivation(z)
        dx = np.dot(dy, self.w)
        return dx, dy
        
    def updateWeights(self, dy):
        g = np.dot(np.atleast_2d(dy).T, np.atleast_2d(self.x))
        self.t += 1
        self.m = 0.9*self.m + (1-0.9)*g
        self.v = 0.999*self.v + (1-0.999)*g**2
        bm = self.m / (1 - 0.9**self.t)
        bv = self.m / (1 - 0.999**self.t)
        self.w += (0.001 / (bv**(0.5) + 10e-8))*bm

