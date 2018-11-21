import random

import numpy as np
import tensorflow as tf

from typing import List
from tensorflow import Tensor


class CompressibleFeedForwardNeuronalNetwork:
    inputs: int
    outputs: int
    x: Tensor
    y: Tensor
    w: List[Tensor]
    b: List[Tensor]
    m: Tensor
    predict: int
    updates: any
    session: tf.Session
    compressed: bool = False
    learning_rate: float

    def __init__(self, layers: List[int], learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.inputs = layers[0]
        self.outputs = layers[len(layers) - 1]

        self.x = tf.placeholder(tf.float32, shape=(self.inputs, None))
        self.y = tf.placeholder(tf.float32, shape=(self.outputs, None))

        self.w = [tf.Variable(tf.random_normal(shape=(layers[i], layers[i - 1]))) for i in range(1, len(layers))]
        self.b = [tf.Variable(tf.random_normal(shape=(layers[i], 1))) for i in range(1, len(layers))]

        a = self.x
        for i in range(len(self.w)):
            a = tf.nn.sigmoid(tf.add(tf.matmul(self.w[i], a), self.b[i]))

        yhat = a

        self.predict = tf.argmax(yhat, axis=0)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=yhat, labels=self.y))
        self.updates = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def feed_forward(self, x) -> int:
        return self.sess.run(self.predict, feed_dict={self.x: x})

    def train(self, x, y, epochs, batch_size, test_x=None, test_y=None) -> None:
        indices = list(range(x.shape[1]))
        for i in range(epochs):
            print("Epoch %d/%d" % (i + 1, epochs))
            random.shuffle(indices)
            for mini_batch in [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]:
                self.train_batch(x[:, mini_batch], y[:, mini_batch])

            if test_x is not None and test_y is not None:
                print("Accuracy: %f%%" % (self.evaluate(test_x, test_y) * 100))

    def train_batch(self, x, y) -> None:
        self.sess.run(self.updates, feed_dict={self.x: x, self.y: y})

    def evaluate(self, x, y) -> float:
        batch_size = x.shape[1]
        results = self.feed_forward(x)
        right = sum(np.equal(results, np.argmax(y, 0)))
        return right / float(batch_size)

    def compress_network(self) -> None:
        w1 = [self.homogenize_matrix(m) for m in self.w]
        b1 = [self.homogenize_vector(v) for v in self.b]

        s = tf.Variable(tf.eye(self.inputs + 1))

        for i in range(len(w1)):
            s = tf.nn.sigmoid(tf.matmul(tf.matmul(s, w1[i]), b1[i]))

        self.m = s

        x = self.homogenize_vector(self.x)

        yhat = tf.slice(tf.matmul(x, self.m), [0], [self.outputs])
        self.predict = tf.argmax(yhat, axis=1)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=yhat, labels=self.y))
        self.updates = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        self.compressed = True

    def __del__(self) -> None:
        self.sess.close()

    def homogenize_matrix(self, m: Tensor) -> Tensor:
        vlen = m.get_shape().as_list()[1]
        vec = np.zeros(shape=(1, vlen))
        m = tf.concat([m, tf.constant(vec, dtype=tf.float32)], 0)
        vlen = m.get_shape().as_list()[0]
        vec = np.zeros(shape=(vlen, 1))
        vec[vlen - 1][0] = 1
        return tf.concat([m, tf.constant(vec, dtype=tf.float32)], 1)

    def homogenize_vector(self, v: Tensor) -> Tensor:
        vec = np.array([[1]])
        return tf.concat([v, tf.constant(vec, dtype=tf.float32)], 1)
