import math
from typing import List

import numpy as np

from ai import NeuronalNetwork
from ai.HomogenousFunctions import homogenize_matrix, homogenize_translation_vector, homogenize_vector


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class CompressableNeuronalNetwork(NeuronalNetwork):
    M: np.ndarray
    R: np.ndarray
    a: any

    def __init__(self, w: List[np.ndarray], b: List[np.ndarray]):
        w = [homogenize_matrix(m) for m in w]
        b = [homogenize_translation_vector(v) for v in b]

        self.a = np.vectorize(sigmoid)

        m = np.identity(w[0].shape[0])

        for l in zip(w, b):
            m = self.a(np.matmul(np.matmul(m, l[0]), l[1]))

        self.M = m
        self.R = np.linalg.pinv(self.M)

    def get_label_count(self) -> int:
        return self.R.shape[1]

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        x = homogenize_vector(x)
        x = np.matmul(x, self.M)
        return x[:, :-1]

    def feed_backwards(self, x: np.ndarray) -> np.ndarray:
        x = homogenize_vector(x)
        x = np.matmul(x, self.R)
        return x[:, :-1]
