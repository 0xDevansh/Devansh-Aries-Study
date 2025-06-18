import numpy as np
import mnist_loader
from random import shuffle

class Network:
    def __init__(self, layers, learning_rate=0.01, lamda = 1, batch_size=100):
        self.layers = layers
        self.n_layers = len(layers)
        self.w = []
        self.b = []
        self.init_weights_biases()
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.batch_size = batch_size

    def init_weights_biases(self):
        """Initialise w and b with std dev as 1/sqrt(no. of inputs of that layer)"""
        for prev, n_next in zip(self.layers, self.layers[1:]):
            self.w.append(np.random.randn(n_next, prev) / np.sqrt(prev))
            self.b.append(np.random.randn(n_next, 1))
    
    def test_loss(self, test_data):
        return sum([(y - self.feedforward(x)[0])[0] ** 2 for x, y in test_data]) / len(test_data)
    
    def feedforward(self, x): # feedforward a single input x (784, 1)
        zs = []
        acts = [x]
        for w, b in zip(self.w, self.b):
            z = w @ x + b
            zs.append(z)
            x = self.sigmoid(z)
            acts.append(x)
        return x, acts, zs
    
    def train_network(self, train_data, test_data):
        # split into random batches
        for i in range(10000):
            idx = np.arange(len(train_data))
            np.random.shuffle(idx)
            for j in range(0, len(train_data), self.batch_size):
                batch = [train_data[k] for k in idx[j:j+self.batch_size]]
                self.process_batch(batch)
            
            print(f'{i}:  {self.test_loss(train_data), self.test_loss(test_data)}')

    def process_batch(self, data):
        # process one batch and return del_w and del_b
        del_w = [np.zeros(w.shape) for w in self.w]
        del_b = [np.zeros(b.shape) for b in self.b]
        for (x, y) in data:
            nabla_w, nabla_b = self.backpropagate(x, y)
            # print_shapes(del_w)
            # print_shapes(nabla_w)
            del_w = [dw + nw for dw, nw in zip(del_w, nabla_w)]
            del_b = [db + nb for db, nb in zip(del_b, nabla_b)]
            # del_w += nabla_w
            # del_b += nabla_b
        
        # update weights and biases
        self.w = [w + dw / self.batch_size * self.learning_rate for w, dw in zip(self.w, del_w)]
        self.b = [b + db / self.batch_size * self.learning_rate for b, db in zip(self.b, del_b)]

    def backpropagate(self, x, y):
        del_w = [0] * len(self.w)
        del_b = [0] * len(self.b)
        pred, acts, zs = self.feedforward(x)

        delta_L = (pred - y) * self.sigmoid_derivative(zs[-1])
        deltas = [0] * self.n_layers
        deltas[-1] = delta_L
        del_b[-1] = delta_L
        del_w[-1] = delta_L @ acts[-2].T


        for i in range(2, self.n_layers):
            delta = (self.w[-i+1].transpose() @ deltas[-i+1]) * self.sigmoid_derivative(zs[-i])
            deltas[-i] = delta
            del_b[-i] = delta
            del_w[-i] = delta @ acts[-i-1].T
        return del_w, del_b
    
    def y_value_to_vector(self, y):
        return np.array([1 if y == i else 0 for i in range(10)]).transpose()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

def print_shapes(arr):
    print(*[v.shape for v in arr])
    
if __name__ == '__main__':
    # test mnist loader
    train_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10], batch_size=10, learning_rate=0.0001, lamda=1)
    net.train_network(list(train_data), list(test_data))