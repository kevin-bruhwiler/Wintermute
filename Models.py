from FCNeuralNetwork import DenseNeuralNetwork
import numpy as np

class Model:

    def __init__(self):
        self.layers = []

    def addFC(self, input_size, output_size, activation, d='d'):
        if input_size < 1 or output_size < 1:
            raise Exception("Invalid layer size. Layer sizes must be positive")
        if d == 'd':
            d = d + activation
        self.layers.append(DenseNeuralNetwork(input_size, output_size, activation, d))

    def predict(self, x):
        for i, layer in enumerate(self.layers):
            layer.x = x
            x = layer.forwardProp()
        return x

    def predict_train(self, x):
        x_states = []
        for i, layer in enumerate(self.layers):
            x_states.append(x)
            layer.x = x
            x = layer.forwardProp()
        return x, x_states

    def train(self, x_train, y_train, batch_size=32, iterations=1, verbose=1):
        if x_train[0] != y_train[0] and len(x_train.shape)>2:
            raise Exception("Mismatched training data: ", x_train[0], "inputs provided, ",
                            y_train[0], "outputs provided")
        batch = []
        xs = []
        for iteration in range(iterations):
            for i, x in enumerate(x_train):
                x, x_states = self.predict_train(x)
                batch.append(x)
                for y in x_states:
                    for z in y:
                        xs.append(z)
                if len(batch) == batch_size:
                    error = ((np.asarray(batch)-y_train[i-batch_size+1:i+1])*abs(np.asarray(batch)-y_train[i-batch_size+1:i+1]))/batch_size
                    self.layers[-1].x = xs[-1]
                    dx, dy = self.layers[-1].trainNetwork(error)
                    self.layers[-1].updateWeights(dy)
                    for l in range(2, len(self.layers)):
                        self.layers[-l].x = xs[-l]
                        dx, dy = self.layers[-l].trainNetwork(dx)
                        self.layers[-l].updateWeights(dy)
                    batch = []
            if verbose == 1:
                print('Iteration: ', iteration, ' Error: ', np.sum(error)) 
                    
