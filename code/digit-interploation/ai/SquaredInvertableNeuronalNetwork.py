import math
from typing import List

import numpy as np

from ai import NeuronalNetwork
from ai.HomogenousFunctions import homogenize_matrix, homogenize_translation_vector, homogenize_vector


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_inverse(x):
    return math.log(1/x - 1)


class SquaredInvertableNeuronalNetwork(NeuronalNetwork):
    M: List[np.ndarray]
    R: List[np.ndarray]
    a: any
    ai: any

    def __init__(self, w: List[np.ndarray], b: List[np.ndarray]):
        w = [homogenize_matrix(m) for m in w]
        b = [homogenize_translation_vector(v) for v in b]

        self.a = np.vectorize(sigmoid)
        self.ai = np.vectorize(sigmoid_inverse)

        self.M = [np.matmul(w[i], b[i]) for i in range(len(w))]
        self.R = list(reversed([np.linalg.inv(m) for m in self.M]))

    def get_label_count(self) -> int:
        return 10

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        x = homogenize_vector(x)
        for m in self.M:
            x = np.matmul(x, m)
            x[:, :-1] = self.a(np.clip(x[:, :-1], -500, 500))
        return x[:, :-1]

    def feed_backwards(self, x: np.ndarray) -> np.ndarray:
        v = np.zeros((1, 28 * 28))
        v[:, 0:10] = x
        x = homogenize_vector(v)
        for m in self.R:
            x[:, :-1] = self.ai(np.clip(x[:, :-1], 0.000000001, 0.999999999))
            x = np.matmul(x, m)
        return x[:, :-1]

    def predict(self, x: np.ndarray):
        return np.argmax(self.feed_forward(x)[:, 0:10], axis=1)
