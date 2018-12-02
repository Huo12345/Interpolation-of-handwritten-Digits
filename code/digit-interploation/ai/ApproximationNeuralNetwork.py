from typing import List

import numpy as np
import tensorflow as tf


class ApproximationNeuralNetwork:

    def __init__(self, w: List[np.ndarray], b: List[np.ndarray], learning_rate: float):
        self.o = tf.Variable(tf.random_normal(shape=(1, w[0].shape[0])))
        W = [tf.constant(m) for m in w]
        B = [tf.constant(m) for m in b]
        self.i = tf.placeholder(tf.float32, shape=(1, w[len(w) - 1].shape[1]))

        a = self.o
        for i in range(len(W)):
            a = tf.nn.sigmoid(tf.add(tf.matmul(a, W[i]), B[i]))

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=a, labels=self.i))
        self.updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)

        self.rest = tf.assign(self.o, tf.random_normal(shape=(1, w[0].shape[0])));

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def approximate(self, target: np.ndarray) -> np.ndarray:
        self.sess.run(self.rest)
        for i in range(100):
            for _ in range(100):
                self.sess.run(self.updates, feed_dict={self.i: target})
            cost = self.sess.run(self.cost, feed_dict={self.i: target})
            print("Epoch %d: %f" % (i, cost))
        return self.sess.run(self.o)
