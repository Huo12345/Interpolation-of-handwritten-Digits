import numpy as np

from ai import MnistNetwork
from ui import disp_image

network = MnistNetwork([100, 50], 0.1, 'C:\\Users\\b-jaq_6xxtrb4\\Downloads')

train_data = network.load_train_data()
test_data = network.load_test_data()

for i in range(3):
    disp_image(train_data[0][i], 'Train data: %d' % np.argmax(train_data[1][i]))

for i in range(3):
    disp_image(test_data[0][i], 'Test data: %d' % np.argmax(test_data[1][i]))

network.train(100, 100)