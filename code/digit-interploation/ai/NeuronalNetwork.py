from abc import ABC, abstractmethod

import numpy as np


class NeuronalNetwork(ABC):

    @abstractmethod
    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_label_count(self) -> int:
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.feed_forward(x), axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        r = self.predict(x)
        right = sum(np.equal(r, np.argmax(y, axis=1)))
        return right / float(len(r))

    def confusion_matrix(self, x, y) -> np.ndarray:
        results = self.predict(x)
        labels = np.argmax(y, axis=1)
        conf = np.zeros((self.get_label_count(), self.get_label_count()), dtype=np.int)
        for x in zip(labels, results):
            conf[x] = conf[x] + 1
        return conf
