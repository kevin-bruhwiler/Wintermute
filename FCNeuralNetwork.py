import numpy as np

class DenseNeuralNetwork:
    
    def __init__(self, xs, ys):
        self.x = np.zeros(xs)
        self.xs = self.x.shape[0]
        self.y = np.zeros(ys)
        self.ys = self.y.shape[0]
        self.w = np.random.rand(ys, xs)
        self.m = np.zeros_like(self.w)
        self.v = np.zeros_like(self.w)
        self.t = 0
        self.accG = np.zeros_like(self.w)
        
    def relu(self, a):
        return np.maximum(a, 0)
        
    def drelu(self, a):
        for i in range(len(a)):
            if a[i] > 0:
                a[i] = 1
            else:
                a[i] = 0 
        return a
    
    def getWeights(self):
        return self.w
        
    def forwardProp(self):
        a = np.dot(self.w, self.x.T)      
        return self.relu(a).T
    
    def loss(self):
        return np.sum(self.forwardProp() - self.y)

    def cost(self):
        return self.forwardProp() - self.y
        
    def trainNetwork(self, z):
        dy = self.drelu(z)
        dx = np.dot(dy, self.w) * self.drelu(self.x)
        return dx, dy

    def accumulate(self, dy):
        self.accG += np.dot(np.atleast_2d(dy).T, np.atleast_2d(self.x))
        
    def updateWeights(self, batch_size):
        g = self.accG / batch_size
        self.accG *= 0
        self.t += 1
        self.m = 0.9*self.m + (1-0.9)*g
        self.v = 0.999*self.v + (1-0.999)*g**2
        bm = self.m / (1 - 0.9**self.t)
        bv = self.m / (1 - 0.999**self.t)
        self.w -= (0.001 / (bv**(0.5) + 10e-8))*bm
