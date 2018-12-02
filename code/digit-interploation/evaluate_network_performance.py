from ai import *

work_dir = '.\\tmp'
out_dir = '.\\out'

accuracy = []
confusion_matrix = []
sq_accuracy = []
sq_confusion_matrix = []
c_accuracy = []
c_confusion_matrix = []
csq_accuracy = []
csq_confusion_matrix = []
i_accuracy = []
i_confusion_matrix = []
isq_accuracy = []
isq_confusion_matrix = []

network = MnistNetwork([100, 50], 0.1, work_dir)

train_data = network.load_train_data()
test_data = network.load_test_data()

best_network = None
best_sq_network = None

best_acc = 0
sample_size = 20

for i in range(sample_size):
    print("Iteration %d of %d" % (i, sample_size))
    network = MnistNetwork([100, 50], 0.1, work_dir)
    acc = network.train(100, 100)
    accuracy.append(acc)
    confusion_matrix.append(network.confusion_matrix(test_data[0], test_data[1]))
    c_network = CompressableNeuronalNetwork([network.network.sess.run(t) for t in network.network.w],
                                            [network.network.sess.run(t) for t in network.network.b])
    c_accuracy.append(c_network.evaluate(test_data[0], test_data[1]))
    c_confusion_matrix.append(c_network.confusion_matrix(test_data[0], test_data[1]))
    i_network = InvertableNeuronalNetwork([network.network.sess.run(t) for t in network.network.w],
                                          [network.network.sess.run(t) for t in network.network.b])
    i_accuracy.append(i_network.evaluate(test_data[0], test_data[1]))
    i_confusion_matrix.append(i_network.confusion_matrix(test_data[0], test_data[1]))
    if acc > best_acc:
        best_acc = acc
        best_network = network

best_acc = 0
for i in range(sample_size):
    print("Iteration %d of %d" % (i, sample_size))
    network = MnistNetwork([28 * 28], 0.1, work_dir)
    acc = network.train(100, 100)
    sq_accuracy.append(acc)
    sq_confusion_matrix.append(network.confusion_matrix(test_data[0], test_data[1]))
    c_network = CompressableNeuronalNetwork([network.network.sess.run(t) for t in network.network.w],
                                            [network.network.sess.run(t) for t in network.network.b])
    csq_accuracy.append(c_network.evaluate(test_data[0], test_data[1]))
    csq_confusion_matrix.append(c_network.confusion_matrix(test_data[0], test_data[1]))
    i_network = InvertableNeuronalNetwork([network.network.sess.run(t) for t in network.network.w],
                                          [network.network.sess.run(t) for t in network.network.b])
    isq_accuracy.append(i_network.evaluate(test_data[0], test_data[1]))
    isq_confusion_matrix.append(i_network.confusion_matrix(test_data[0], test_data[1]))
    if acc > best_acc:
        best_acc = acc
        best_sq_network = network
