import numpy as np
import matplotlib.pyplot as plt

from Network import Network

import mnist

x_train, t_train, x_test, t_test = mnist.load()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

net = Network([784, 16, 16, 10], 0.1)

total_epochs = 10
batch_size = 32

total_batches = len(x_train) / batch_size

epochs_done = 0

performances = []


def calc_performance():
    errors = 0

    for x_test_item, t_test_value in zip(x_test, t_test):
        res = np.argmax(net.predict(np.array([x_test_item]).T))

        if res != t_test_value:
            errors += 1

    performance = (len(t_test) - errors) / len(t_test) * 100

    return performance


performances.append(calc_performance())

while epochs_done < total_epochs:
    trainings_done = 0

    while trainings_done < total_batches:
        input_matrix = x_train[trainings_done:trainings_done + batch_size].T
        expected_values = t_train[trainings_done:trainings_done + batch_size]

        expected_matrix = np.zeros((10, input_matrix.shape[1]),
                                   dtype=np.float32)

        for index, value in enumerate(expected_values):
            expected_matrix[value, index] = 1.0

        net.back_propagate(input_matrix, expected_matrix)

        trainings_done += batch_size

    print('Epoch ' + str(epochs_done) + ' done')

    performances.append(calc_performance())

    epochs_done += 1

plt.plot(np.arange(total_epochs + 1), performances, marker='o')
plt.show()
