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