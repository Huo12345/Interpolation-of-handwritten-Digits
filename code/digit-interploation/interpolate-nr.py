import numpy as np

from ai import MnistNetwork, InvertableNeuronalNetwork, ApproximationNeuralNetwork, CompressableNeuronalNetwork
from ui import disp_image

work_dir = '.\\tmp'
out_dir = '.\\out'

network = MnistNetwork([100, 50], 0.1, work_dir)

train_data = network.load_train_data()
test_data = network.load_test_data()

for i in range(3):
    disp_image(train_data[0][i], 'Train data: %d' % np.argmax(train_data[1][i]))

for i in range(3):
    disp_image(test_data[0][i], 'Test data: %d' % np.argmax(test_data[1][i]))

acc = network.train(100, 100)
print("Network accuracy: %f%%" % (acc * 100))
print(network.network.confusion_matrix(test_data[0], test_data[1]))

c_network = CompressableNeuronalNetwork([network.network.sess.run(t) for t in network.network.w],
                                        [network.network.sess.run(t) for t in network.network.b])
acc = c_network.evaluate(test_data[0], test_data[1])
print("Compressed network accuracy: %f%%" % (acc * 100))

i_network = InvertableNeuronalNetwork([network.network.sess.run(t) for t in network.network.w],
                                        [network.network.sess.run(t) for t in network.network.b])
acc = i_network.evaluate(test_data[0], test_data[1])
print("Invertable network accuracy: %f%%" % (acc * 100))

a_network = ApproximationNeuralNetwork([network.network.sess.run(t) for t in network.network.w],
                                       [network.network.sess.run(t) for t in network.network.b], 1.0)


# v = np.zeros((1, 10))
# v[0][0] = 1
approximated_image = a_network.approximate(network.network.feed_forward(train_data[0][0].reshape(1, 28*28)))
disp_image(approximated_image)
print(network.network.feed_forward(approximated_image))

# c_network = CompressedNeuronalNetwork([network.network.sess.run(t) for t in network.network.w],
#                                       [network.network.sess.run(t) for t in network.network.b])

# acc = c_network.evaluate(test_data[0], test_data[1])
# print("Compressed network accuracy: %f%%" % (acc * 100))
#
# v = np.zeros((1, 10))
# v[0][0] = 1
#
# disp_image(c_network.feed_backwards(c_network.feed_forward(test_data[0][0].reshape(1, network.image_size))))
# disp_image(c_network.feed_backwards(v))
