import math
from typing import List

import numpy as np


def homogenize_matrix(m: np.ndarray) -> np.ndarray:
    h = np.zeros((m.shape[0] + 1, m.shape[1] + 1))
    h[:-1, :-1] = m
    h[m.shape[0]][m.shape[1]] = 1
    return h


def homogenize_translation_vector(v: np.ndarray) -> np.ndarray:
    n = v.shape[1] + 1
    h = np.identity(n)
    h[n - 1, :-1] = v
    return h


def homogenize_vector(v: np.ndarray) -> np.ndarray:
    h = np.ones((v.shape[0], v.shape[1] + 1))
    h[:, :-1] = v
    return h


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_inverse(x):
    return -math.log(1/x - 1)


class CompressedNeuronalNetwork:
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

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        x = homogenize_vector(x)
        for m in self.M:
            x = np.matmul(x, m)
            x[:, :-1] = self.a(np.clip(x[:, :-1], -50, 50))
        return x[:, :-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.feed_forward(x), axis=1)

    def feed_backwards(self, x: np.ndarray) -> np.ndarray:
        x = homogenize_vector(x)
        for m in self.R:
            x[:, :-1] = self.ai(np.clip(x[:, :-1], 1e-18, 0.9999999))
            x = np.matmul(x, m)
        return x[:, :-1]

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        r = self.predict(x)
        right = sum(np.equal(r, np.argmax(y, axis=1)))
        return right / float(len(r))
