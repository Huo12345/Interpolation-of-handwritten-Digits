import numpy as np

from ai import MnistNetwork
from ui import disp_image, save_image


def softmax_cross_entropy(x, y):
    e = np.exp(x)
    p = e / np.sum(e)
    return -np.sum(y * np.log(p))

work_dir = '.\\tmp'
out_dir = '.\\out'

network = MnistNetwork([100, 50], 0.1, work_dir)

train_data = network.load_train_data()
test_data = network.load_test_data()

for i in range(3):
    disp_image(train_data[0][i], 'Check train data, expected value: %d' % np.argmax(train_data[1][i]))

for i in range(3):
    disp_image(test_data[0][i], 'Check test data, expected value: %d' % np.argmax(test_data[1][i]))

acc = network.train(100, 100)
print("Network accuracy: %f%%" % (acc * 100))
print(network.network.confusion_matrix(test_data[0], test_data[1]))

predictions = network.feed_forward(test_data[0])

top = [(10, -1) for i in range(10)]
flop = [(0, -1) for i in range(10)]
top_false = [(10, -1) for i in range(10)]

for i in range(len(predictions)):
    dig = int(np.argmax(test_data[1][i]))
    err = softmax_cross_entropy(predictions[i], test_data[1][i])
    if np.argmax(predictions[i]) != np.argmax(test_data[1][i]):
        if err < top_false[dig][0]:
            top_false[dig] = (err, i)
    else:
        if err < top[dig][0]:
            top[dig] = (err, i)
        if err > flop[dig][0]:
            flop[dig] = (err, i)

run = 2

for dig in range(10):
    (err, i) = top[dig]
    save_image("%s\\top-flop\\clear_%d_%d.png" % (out_dir, dig, run), test_data[0][i],
               "Klare %d, Fehler: %f" % (dig, err))
    print("Top %d, err: %f: vec: %s" % (dig, err, ["%.2f" % f for f in predictions[i]]))

for dig in range(10):
    (err, i) = flop[dig]
    p = predictions[i]
    p[np.argmax(p)] = 0
    almost = int(np.argmax(p))
    save_image("%s\\top-flop\\barely_%d_%d.png" % (out_dir, dig, run), test_data[0][i],
               "Knappe %d, Fast: %d, Fehler: %f" % (dig, almost, err))

    print("Flop %d, almost: %d, err: %f: vec: %s" % (dig, almost, err, ["%.2f" % f for f in predictions[i]]))


for dig in range(10):
    (err, i) = top_false[dig]
    if i == -1:
        continue
    save_image("%s\\top-flop\\flop_%d_%d.png" % (out_dir, dig, run), test_data[0][i],
               "Knapp nicht %d, erkannt %d, Fehler: %f" % (dig, int(np.argmax(predictions[i])), err))
    print("Barely false %d, detected: %d, err: %f: vec: %s" % (dig, int(np.argmax(predictions[i])), err,
                                                               ["%.2f" % f for f in predictions[i]]))
