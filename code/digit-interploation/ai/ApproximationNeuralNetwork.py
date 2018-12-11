from typing import List, Tuple

import numpy as np
import tensorflow as tf


class ApproximationNeuralNetwork:

    input_size: int

    def __init__(self, w: List[np.ndarray], b: List[np.ndarray], learning_rate: float):
        self.input_size = w[0].shape[0]
        self.o = tf.Variable(tf.random_normal(shape=(1, self.input_size)))
        W = [tf.constant(m) for m in w]
        B = [tf.constant(m) for m in b]
        self.i = tf.placeholder(tf.float32, shape=(1, w[len(w) - 1].shape[1]))
        self.start = tf.placeholder(tf.float32, shape=(1, self.input_size))

        a = self.o
        for i in range(len(W)):
            a = tf.nn.sigmoid(tf.add(tf.matmul(a, W[i]), B[i]))

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=a, labels=self.i))
        self.updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)

        self.rest = tf.assign(self.o, self.start)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def approximate(self, target: np.ndarray, start: np.ndarray = None) -> Tuple[np.ndarray, float]:
        if start is None:
            start = np.random.normal(0, 1, self.input_size).reshape(1, self.input_size)
        self.sess.run(self.rest, feed_dict={self.start: start})
        for i in range(100):
            for _ in range(100):
                self.sess.run(self.updates, feed_dict={self.i: target})
        return self.get_status(target)

    def get_status(self, target: np.ndarray):
        return self.sess.run(self.o), self.sess.run(self.cost, feed_dict={self.i: target})
