import numpy as np

from activations import tanh, d_tanh
from loss_functions import squared_error, d_squared_error


class Network:
    def __init__(self, structure, lr):
        self.weights = [
            np.random.rand(j, i) * 2 - 1
            for j, i in zip(structure[1:], structure)
        ]
        self.biases = [np.random.rand(i, 1) * 2 - 1 for i in structure[1:]]

        self.lr = lr

    def predict(self, input, for_training=False):
        result = input

        z = []
        a = [input]

        for weight, bias in zip(self.weights, self.biases):
            result = weight @ result + bias

            if for_training == True:
                a.append(result)

            result = tanh(result)

            if for_training == True:
                z.append(result)

        if for_training == True:
            return result, z, a

        return result

    def back_propagate(self, input, expected):
        result, z, a = self.predict(input, True)

        dCost_dAct = d_squared_error(result, expected)

        for index in reversed(range(len(self.weights))):
            dAct_dZ = d_tanh(z[index]) * dCost_dAct

            dAct_dZ_avgd = dAct_dZ / dAct_dZ.shape[1]

            dCost_dW = dAct_dZ_avgd @ a[index].T
            dCost_dB = np.sum(dAct_dZ_avgd, axis=1, keepdims=True)

            dCost_dAct = np.dot(self.weights[index].T, dAct_dZ)

            self.weights[index] -= self.lr * dCost_dW
            self.biases[index] -= self.lr * dCost_dB

        return result

    # def save(self, file_name):
    #     np.savez_compressed(file_name, a=test_array, b=test_vector)
