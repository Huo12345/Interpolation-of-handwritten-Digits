import random

import numpy as np
import tensorflow as tf

from typing import List
from tensorflow import Tensor


class FeedForwardNeuronalNetwork:
    inputs: int
    outputs: int
    x: Tensor
    y: Tensor
    predict: int
    updates: any
    session: tf.Session

    def __init__(self, layers: List[int], learning_rate: float) -> None:
        self.inputs = layers[0]
        self.outputs = layers[len(layers) - 1]

        self.x = tf.placeholder(tf.float32, shape=(None, self.inputs))
        self.y = tf.placeholder(tf.float32, shape=(None, self.outputs))

        w = [tf.Variable(tf.random_normal(shape=(layers[i - 1], layers[i]))) for i in range(1, len(layers))]
        b = [tf.Variable(tf.random_normal(shape=(1, layers[i]))) for i in range(1, len(layers))]

        a = self.x
        for i in range(len(w)):
            a = tf.nn.sigmoid(tf.add(tf.matmul(a, w[i]), b[i]))

        yhat = a

        self.predict = tf.argmax(yhat, axis=1)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=yhat, labels=self.y))
        self.updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def feed_forward(self, x) -> int:
        return self.sess.run(self.predict, feed_dict={self.x: x})

    def train(self, x, y, epochs, batch_size, test_x=None, test_y=None) -> None:
        indices = [i for i in range(len(x))]
        for i in range(epochs):
            print("Epoch %d/%d" % (i + 1, epochs))
            random.shuffle(indices)
            for mini_batch in [indices[i:i + batch_size] for i in range(0, len(x), batch_size)]:
                self.train_batch([x[i] for i in mini_batch], [y[i] for i in mini_batch])

            if test_x is not None and test_y is not None:
                print("Accuracy: %f%%" % (self.evaluate(test_x, test_y) * 100))

    def train_batch(self, x, y) -> None:
        self.sess.run(self.updates, feed_dict={self.x: x, self.y: y})

    def evaluate(self, x, y) -> float:
        batch_size = len(x)
        results = self.feed_forward(x)
        right = sum([1 if results[i] == np.argmax(y[i]) else 0 for i in range(batch_size)])

        return right / float(batch_size)

    def __del__(self) -> None:
        self.sess.close()