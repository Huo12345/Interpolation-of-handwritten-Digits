import random

import numpy as np
import tensorflow as tf

from typing import List
from tensorflow import Tensor

from . import NeuronalNetwork


class FeedForwardNeuronalNetwork(NeuronalNetwork):
    inputs: int
    outputs: int
    x: Tensor
    y: Tensor
    w: List[Tensor]
    b: List[Tensor]
    argmax: int
    updates: any
    session: tf.Session

    def __init__(self, layers: List[int], learning_rate: float) -> None:
        self.inputs = layers[0]
        self.outputs = layers[len(layers) - 1]

        self.x = tf.placeholder(tf.float32, shape=(None, self.inputs))
        self.y = tf.placeholder(tf.float32, shape=(None, self.outputs))

        self.w = [tf.Variable(tf.random_normal(shape=(layers[i - 1], layers[i]))) for i in range(1, len(layers))]
        self.b = [tf.Variable(tf.random_normal(shape=(1, layers[i]))) for i in range(1, len(layers))]

        a = self.x
        for i in range(len(self.w)):
            a = tf.nn.sigmoid(tf.add(tf.matmul(a, self.w[i]), self.b[i]))

        self.yhat = a

        self.argmax = tf.argmax(self.yhat, axis=1)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.yhat, labels=self.y))
        self.updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_label_count(self) -> int:
        return self.outputs

    def feed_forward(self, x) -> np.ndarray:
        return self.sess.run(self.yhat, feed_dict={self.x: x})

    def predict(self, x) -> np.ndarray:
        return self.sess.run(self.argmax, feed_dict={self.x: x})

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

    def __del__(self) -> None:
        self.sess.close()
