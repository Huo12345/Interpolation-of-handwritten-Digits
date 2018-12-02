import gzip
import math
import os
import urllib.request
from typing import List, Tuple

import numpy as np

from . import FeedForwardNeuronalNetwork, NeuronalNetwork


class MnistNetwork(NeuronalNetwork):
    layers: List[int]
    learning_rate: float
    work_dir: str

    network: FeedForwardNeuronalNetwork
    image_size: int
    train_data: np.ndarray
    test_data: np.ndarray

    MNIST_URL: str = 'http://yann.lecun.com/exdb/mnist/'

    def __init__(self, layers: List[int], learning_rate: float, work_dir: str) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.work_dir = work_dir
        self.network = None
        self.image_size = 0
        self.train_data = None
        self.test_data = None

    def train(self, epochs: int, batch_size: int) -> float:
        if self.train_data is None:
            self.train_data = self.load_train_data()
        if self.test_data is None:
            self.test_data = self.load_test_data()

        if self.network is None:
            self.network = FeedForwardNeuronalNetwork([self.image_size] + self.layers + [10], self.learning_rate)

        train_data, eval_data = self.split_data_set(self.train_data[0], 0.9)
        train_label, eval_label = self.split_data_set(self.train_data[1], 0.9)

        self.network.train(train_data, train_label, epochs, batch_size, eval_data, eval_label)
        return self.network.evaluate(self.test_data[0], self.test_data[1])

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        return self.network.feed_forward(x)

    def get_label_count(self) -> int:
        return self.network.get_label_count()

    def split_data_set(self, data_set: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        size = len(data_set)
        indeces = range(size)
        delim = int(math.floor(size * ratio))
        p1 = np.array([data_set[i] for i in indeces[:delim]])
        p2 = np.array([data_set[i] for i in indeces[delim:]])
        return p1, p2

    def load_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        images = self.extract_images(self.download_file_if_needed('train-images-idx3-ubyte.gz'))
        labels = self.extract_labels(self.download_file_if_needed('train-labels-idx1-ubyte.gz'))
        return images, labels

    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        images = self.extract_images(self.download_file_if_needed('t10k-images-idx3-ubyte.gz'))
        labels = self.extract_labels(self.download_file_if_needed('t10k-labels-idx1-ubyte.gz'))
        return images, labels

    def download_file_if_needed(self, file: str) -> str:
        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)
        file_path = os.path.join(self.work_dir, file)
        if not os.path.exists(file_path):
            print("Downloading " + file)
            urllib.request.urlretrieve(self.MNIST_URL + file, file_path)
        else:
            print(file + " already downloaded")
        return file_path

    def extract_images(self, file: str) -> np.ndarray:
        with gzip.open(file) as bytestream:
            bytestream.read(4)
            nr_of_images = int.from_bytes(bytestream.read(4), byteorder='big', signed=False)
            image_width = int.from_bytes(bytestream.read(4), byteorder='big', signed=False)
            image_height = int.from_bytes(bytestream.read(4), byteorder='big', signed=False)
            self.image_size = image_width * image_height

            buffer = bytestream.read(self.image_size * nr_of_images)
            images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            images = images.reshape(nr_of_images, self.image_size)
            return images

    def extract_labels(self, file: str) -> np.ndarray:
        with gzip.open(file) as bytestream:
            bytestream.read(4)
            nr_of_labels = int.from_bytes(bytestream.read(4), byteorder='big', signed=False)

            buffer = bytestream.read(nr_of_labels)
            labels = np.frombuffer(buffer, dtype=np.uint8)
            return (np.arange(10) == labels[:, None]).astype(np.float32)
