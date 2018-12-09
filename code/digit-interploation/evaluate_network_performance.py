import math
from typing import Tuple

import numpy as np

from ai import *
from ui import save_histogram, save_image, create_confusion_table_matrix


def interpolate_vector(v: np.ndarray, dimen: Tuple[int, int], factor: float) -> np.ndarray:
    m = np.identity(v.shape[1])
    a = math.pi / 4 * factor
    m[dimen[0], dimen[0]] = math.cos(a)
    m[dimen[0], dimen[1]] = math.sin(a)
    m[dimen[1], dimen[0]] = -math.sin(a)
    m[dimen[1], dimen[1]] = math.cos(a)
    return np.matmul(v, m)


work_dir = '.\\tmp'
out_dir = '.\\out'

accuracy = []
c_accuracy = []
c2_accuracy = []
i_accuracy = []

network = MnistNetwork([100, 50], 0.1, work_dir)

train_data = network.load_train_data()
test_data = network.load_test_data()

best_network = None
best_sq_network = None

best_acc = 0
sample_size = 20

labels = list(map(str, range(10)))

for i in range(sample_size):
    print("Iteration %d of %d" % (i + 1, sample_size))
    network = MnistNetwork([100, 50], 0.1, work_dir)
    acc = network.train(100, 100)
    accuracy.append(acc)
    confusion_matrix = network.confusion_matrix(test_data[0], test_data[1])
    create_confusion_table_matrix("%s\confusion_table_network_%d.tex" % (out_dir, i), confusion_matrix, labels,
                                  "Konfusionsmatrix Netzwerk %d" % i, "tbl:confusion_table_network_%d" % i)
    c_network = CompressableNeuronalNetwork([network.network.sess.run(t) for t in network.network.w],
                                            [network.network.sess.run(t) for t in network.network.b])
    c_accuracy.append(c_network.evaluate(test_data[0], test_data[1]))
    c_confusion_matrix = c_network.confusion_matrix(test_data[0], test_data[1])
    create_confusion_table_matrix("%s\confusion_table_compressed_network_%d.tex" % (out_dir, i), c_confusion_matrix,
                                  labels, "Konfusionsmatrix Komprimiertes Netzwerk Art 1 #%d" % i,
                                  "tbl:confusion_table_compressed_network_%d" % i)
    c2_network = CompressableNeuronalNetworkV2([network.network.sess.run(t) for t in network.network.w],
                                               [network.network.sess.run(t) for t in network.network.b])
    c2_accuracy.append(c2_network.evaluate(test_data[0], test_data[1]))
    c2_confusion_matrix = c2_network.confusion_matrix(test_data[0], test_data[1])
    create_confusion_table_matrix("%s\confusion_table_compressed_network_v2_%d.tex" % (out_dir, i), c2_confusion_matrix,
                                  labels, "Konfusionsmatrix Komprimiertes Netzwerk Art 2 #%d" % i,
                                  "tbl:confusion_table_compressed_network_v2_%d" % i)
    i_network = InvertableNeuronalNetwork([network.network.sess.run(t) for t in network.network.w],
                                          [network.network.sess.run(t) for t in network.network.b])
    i_accuracy.append(i_network.evaluate(test_data[0], test_data[1]))
    i_confusion_matrix = i_network.confusion_matrix(test_data[0], test_data[1])
    create_confusion_table_matrix("%s\confusion_table_inverted_network_%d.tex" % (out_dir, i), i_confusion_matrix,
                                  labels, "Konfusionsmatrix Invertiertes Netzwerk %d" % i,
                                  "tbl:confusion_table_inverted_network_%d" % i)
    if acc > best_acc:
        best_acc = acc
        best_network = network

save_histogram("%s\histogram_network_accuracy.png" % out_dir, accuracy, "Präzision Erkennungsnetzwerke")
save_histogram("%s\histogram_compressed_network_accuracy.png" % out_dir, c2_accuracy, "Präzision Komprimierte Netzwerke Art 1")
save_histogram("%s\histogram_compressed_network_v2_accuracy.png" % out_dir, c_accuracy, "Präzision Komprimierte Netzwerke Art 2")
save_histogram("%s\histogram_inverted_network_accuracy.png" % out_dir, accuracy, "Präzision Invertierbare Nettzwerke")

sq_accuracy = []
isq_accuracy = []
best_acc = 0
for i in range(sample_size):
    print("Iteration %d of %d" % (i, sample_size))
    network = MnistNetwork([], 0.1, work_dir, squared=True)
    acc = network.train(100, 100)
    sq_accuracy.append(acc)
    sq_confusion_matrix = network.confusion_matrix(test_data[0], test_data[1])
    create_confusion_table_matrix("%s\confusion_table_squared_network_%d.tex" % (out_dir, i), sq_confusion_matrix,
                                  labels, "Konfusionsmatrix Quadratisches Netzwerk %d" % i,
                                  "tbl:confusion_table_squared_network_%d" % i)
    i_network = SquaredInvertableNeuronalNetwork([network.network.sess.run(t) for t in network.network.w],
                                                 [network.network.sess.run(t) for t in network.network.b])
    isq_accuracy.append(i_network.evaluate(test_data[0], test_data[1]))
    isq_confusion_matrix = i_network.confusion_matrix(test_data[0], test_data[1])
    create_confusion_table_matrix("%s\confusion_table_squared_inverted_network_%d.tex" % (out_dir, i),
                                  isq_confusion_matrix, labels,
                                  "Konfusionsmatrix Quadratisches Invertiertes Netzwerk %d" % i,
                                  "tbl:confusion_table_squared_inverted_network_%d" % i)
    if acc > best_acc:
        best_acc = acc
        best_sq_network = network

save_histogram("%s\histogram_squared_network_accuracy.png" % out_dir, sq_accuracy,
               "Präzision Quadratische Erkennungsnetzwerke")
save_histogram("%s\histogram_squared_inverted_network_accuracy.png" % out_dir, accuracy,
               "Präzision Quadratische Invertierbare Nettzwerke")

i_reversed_network = InvertableNeuronalNetwork([best_network.network.sess.run(t) for t in best_network.network.w],
                                               [best_network.network.sess.run(t) for t in best_network.network.b])

a_reversed_network = ApproximationNeuralNetwork([best_network.network.sess.run(t) for t in best_network.network.w],
                                                [best_network.network.sess.run(t) for t in best_network.network.b], 1.0)

sq_i_reversed_network = SquaredInvertableNeuronalNetwork(
    [best_sq_network.network.sess.run(t) for t in best_sq_network.network.w],
    [best_sq_network.network.sess.run(t) for t in best_sq_network.network.b])

approximation_error_ideal = []

for i in range(10):
    v = np.zeros((1, 10))
    v[0, i] = 1
    image = i_reversed_network.feed_backwards(v)
    save_image("%s\ideal_%d_inverted.png" % (out_dir, i), image, "Ideale %d aus Invertierung" % i)
    image = sq_i_reversed_network.feed_backwards(v)
    save_image("%s\ideal_%d_squared_inverted.png" % (out_dir, i), image, "Ideale %d aus Quadratischer Invertierung" % i)
    image, error = a_reversed_network.approximate(v)
    approximation_error_ideal.append(error)
    save_image("%s\ideal_%d_approximated.png" % (out_dir, i), image, "Ideale %d aus Approximierung" % i)

save_histogram("%s\histogram_approximation_error_ideal.png" % out_dir, approximation_error_ideal,
               "Fehler Approximierung idealer Ziffern")

approximation_error_interpolated = []
for i in range(10):
    for j in range(i, 10):
        for k in [0.25, 0.5, 0.75]:
            l = int(k * 100)
            v = np.zeros((1, 10))
            v[0, i] = 1
            v = interpolate_vector(v, (i, j), k)
            image = i_reversed_network.feed_backwards(v)
            save_image("%s\interpolated_%d_%d_%d_inverted.png" % (out_dir, i, j, l), image,
                       "Interpolierte Ziffer %d%% zwischen %d und %d aus Invertierung" % (l, i, j))
            image = sq_i_reversed_network.feed_backwards(v)
            save_image("%s\interpolated_%d_%d_%d_squared_inverted.png" % (out_dir, i, j, l), image,
                       "Interpolierte Ziffer %d%% zwischen %d und %d aus Quadratischer Invertierung" % (l, i, j))
            image, error = a_reversed_network.approximate(v)
            approximation_error_interpolated.append(error)
            save_image("%s\interpolated_%d_%d_%d_approximated.png" % (out_dir, i, j, l), image,
                       "Interpolierte Ziffer %d%% zwischen %d und %d aus Approximierung" % (l, i, j))

save_histogram("%s\histogram_approximation_error_interpolation.png" % out_dir, approximation_error_interpolated,
               "Fehler Approximierung interpolierter Ziffern")
