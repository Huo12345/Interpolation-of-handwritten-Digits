from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def disp_image(image: np.ndarray, title: str = None) -> None:
    make_image(image, title)
    plt.show()


def save_image(file_name: str, image: np.ndarray, title: str = None) -> None:
    make_image(image, title)
    plt.savefig(file_name)
    print("Created %s" % file_name)


def make_image(image: np.ndarray, title: str = None) -> None:
    plt.clf()
    plt.imshow(image.reshape(28, 28), cmap=plt.cm.Greys)
    if title is not None:
        plt.title(title)


def disp_histogram(data: List, topic: str, title: str = None) -> None:
    make_histogram(data, topic, title)
    plt.show()


def save_histogram(file_name: str, data: List, topic: str, title: str = None) -> None:
    make_histogram(data, topic, title)
    plt.savefig(file_name)
    plt.close()
    print("Created %s" % file_name)


def make_histogram(data: List, topic: str, title: str = None) -> None:
    plt.clf()
    plt.hist(data, density=True)

    (mu, sigma) = norm.fit(data)
    x = np.linspace(min(data), max(data), 100)

    y = norm(mu, sigma).pdf(x)
    plt.plot(x, y, 'r', linewidth=2)

    if title is not None:
        plt.title(title)

    plt.xlabel(topic)
    plt.ylabel("Relative Dichte")

    plt.grid(True)
