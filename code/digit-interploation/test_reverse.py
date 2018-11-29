import numpy as np
import matplotlib.pyplot as plt

from ai import CompressedNeuronalNetwork


def disp_image(image: np.ndarray, size: int) -> None:
    plt.imshow(image.reshape(size, size), cmap=plt.cm.Greys)
    plt.show()


layers = 2
size = 5
le = size * size

network = CompressedNeuronalNetwork([np.random.rand(le, le) for _ in range(layers)], [np.random.rand(1, le) for _ in range(layers)])

i = np.random.rand(1, le)

d = network.feed_forward(i)
ir = network.feed_backwards(d)

disp_image(i, size)
disp_image(ir, size)
