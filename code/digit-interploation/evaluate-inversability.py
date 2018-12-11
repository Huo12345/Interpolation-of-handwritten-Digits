from functools import reduce

import matplotlib.pyplot as plt
import numpy as np

from ai import InvertableNeuronalNetwork


def generate_random_matrix(x: int) -> np.ndarray:
    return np.random.normal(0, 1, x * x).reshape(x, x)


def generate_random_vector(x: int) -> np.ndarray:
    return np.random.normal(0, 1, x).reshape(1, x)


def disp(img: np.ndarray) -> None:
    plt.clf()
    plt.imshow(img, cmap=plt.cm.Greys)
    # plt.show()


m = generate_random_vector(25)
disp(m.reshape(5, 5))
n = InvertableNeuronalNetwork([generate_random_matrix(25) for _ in range(3)],
                              [generate_random_vector(25) for _ in range(3)])

disp(n.feed_backwards(n.feed_forward(m)).reshape(5, 5))


res = np.zeros((7, 4))

for i in range(3, 10):
    for j in range(1, 5):
        l = []
        for _ in range(20):
            m = generate_random_vector(i * i)
            n = InvertableNeuronalNetwork([generate_random_matrix(i * i) for _ in range(j)],
                                          [generate_random_vector(i * i) for _ in range(j)])
            x = n.feed_backwards(n.feed_forward(m))
            l.append(np.sum(np.abs(m - x)))
        err = (reduce(lambda x, y: x + y, l) / len(l))
        res[(i - 3, j - 1)] = err / (i * i)

print(res)
