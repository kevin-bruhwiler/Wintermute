import FCNeuralNetwork
from FCNeuralNetwork import DenseNeuralNetwork
import numpy as np

class Model:

    def __init__(self):
        self.layers = []
        self.gradients = []

    def addFC(self, input_size, output_size):
        if input_size < 1 or output_size < 1:
            raise Exception("Invalid layer size. Layer sizes must be positive")
        self.layers.append(DenseNeuralNetwork(input_size, output_size))

    def predict(self, x):
        for i, layer in enumerate(self.layers):
            layer.x = x
            x = layer.forwardProp()
        return x

    def train(self, x_train, y_train, batch_size=32, iterations=1, verbose=1):
        if x_train[0] != y_train[0] and len(x_train.shape)>2:
            raise Exception("Mismatched training data: ", x_train[0], "inputs provided, ",
                            y_train[0], "outputs provided")
        batch_count = 0
        for iteration in range(iterations):
            for i, x in enumerate(x_train):
                y = self.predict(x)
                error = y - y_train[i]
                dx, dy = self.layers[-1].trainNetwork(error)
                if batch_count == batch_size:
                    self.layers[-1].updateWeights(batch_size)
                else:
                    self.layers[-1].accumulate(dy)
                for l in range(2, len(self.layers)):
                    dx, dy = self.layers[-l].trainNetwork(dx)
                    if batch_count == batch_size:
                        self.layers[-l].updateWeights(batch_size)
                    else:
                        self.layers[-l].accumulate(dy)
                if batch_count == batch_size:
                    batch_count = 0
                else:    
                    batch_count += 1
                if verbose == 1:
                    print('Iteration: ', iteration, ' Error: ', np.sum(error)) 
                    
