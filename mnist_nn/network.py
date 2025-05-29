import numpy as np
import mnist_loader

class Network:
    def __init__(self, layers, learning_rate=0.01, lamda = 1):
        self.layers = layers
        self.n_layers = len(layers)
        self.w = []
        self.b = []
        self.init_weights_biases()
        self.learning_rate = learning_rate
        self.lamda = lamda

    def init_weights_biases(self):
        """Initialise w and b with std dev as 1/sqrt(no. of inputs of that layer)"""
        for prev, n_next in zip(self.layers, self.layers[1:]):
            self.w.append(np.random.randn(n_next, prev) / np.sqrt(prev))
            self.b.append(np.random.randn(n_next, 1))

    def feedforward(self, x): # feedforward a single input x (784, 1)
        zs = []
        acts = []
        for w, b in zip(self.w, self.b):
            z = w @ x + b
            zs.append(zs)
            x = self.sigmoid(z)
            acts.append(x)
        return x, acts, zs

    def process_batch(self, data):
        # process one batch and return del_w and del_b
        del_w = [np.zeros(w.shape) for w in self.w]
        del_b = [np.zeros(b.shape) for b in self.b]
        for (x, y) in data:
            nabla_w, nabla_b = backpropagate(x, y)

    def backpropagate(self, x, y):



    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # test mnist loader
    train_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    for x, y in train_data:
        print(x.shape, y.shape)