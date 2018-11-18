import matplotlib.pyplot as plt
import numpy as np


def disp_image(image: np.ndarray, title: str = None) -> None:
    plt.imshow(image.reshape(28, 28), cmap=plt.cm.Greys)
    if title is not None:
        plt.title(title)
    plt.show()
