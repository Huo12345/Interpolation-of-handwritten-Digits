import random

import numpy as np
import tensorflow as tf

from typing import List
from tensorflow import Tensor


def homogenize_matrix(m: Tensor) -> Tensor:
    vlen = m.get_shape().as_list()[1]
    vec = np.zeros(shape=(1, vlen))
    m = tf.concat([m, tf.constant(vec, dtype=tf.float32)], 0)
    vlen = m.get_shape().as_list()[0]
    vec = np.zeros(shape=(vlen, 1))
    vec[vlen - 1][0] = 1
    return tf.concat([m, tf.constant(vec, dtype=tf.float32)], 1)


def homogenize_translation_vector(m: Tensor) -> Tensor:
    vlen = m.get_shape().as_list()[1]
    vec = tf.eye(vlen, dtype=tf.float32)
    m = tf.concat([vec, m], 0)
    vlen = m.get_shape().as_list()[0]
    vec = np.zeros(shape=(vlen, 1))
    vec[vlen - 1][0] = 1
    return tf.concat([m, tf.constant(vec, dtype=tf.float32)], 1)


def homogenize_vector(v: Tensor) -> Tensor:
    vec = np.array([[1]])
    return tf.concat([v, tf.constant(vec, dtype=tf.float32)], 1)


class FeedForwardNeuronalNetwork:
    learning_rate: float
    inputs: int
    outputs: int
    x: Tensor
    y: Tensor
    predict: any
    updates: any

    compress: any
    m: Tensor
    predict_c: any

    session: tf.Session

    def __init__(self, layers: List[int], learning_rate: float) -> None:
        self.learning_rate = learning_rate
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

        self.inti_compress(w, b)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def feed_forward(self, x) -> int:
        return self.sess.run(self.predict, feed_dict={self.x: x})

    def feed_forward_c(self, x: np.ndarray) -> int:
        return self.sess.run(self.predict_c, feed_dict={self.x: x.reshape(1, x.shape[0])})

    def train(self, x, y, epochs, batch_size, test_x=None, test_y=None) -> None:
        indices = [i for i in range(len(x))]
        for i in range(epochs):
            print("Epoch %d/%d" % (i + 1, epochs))
            random.shuffle(indices)
            for mini_batch in [indices[i:i + batch_size] for i in range(0, len(x), batch_size)]:
                self.train_batch([x[i] for i in mini_batch], [y[i] for i in mini_batch])

            if test_x is not None and test_y is not None:
                print("Accuracy: %f%%" % (self.evaluate(test_x, test_y) * 100))

    def compress_network(self) -> None:
        self.sess.run(self.compress)

    def train_batch(self, x, y) -> None:
        self.sess.run(self.updates, feed_dict={self.x: x, self.y: y})

    def evaluate(self, x, y) -> float:
        batch_size = len(x)
        results = self.feed_forward(x)
        right = sum([1 if results[i] == np.argmax(y[i]) else 0 for i in range(batch_size)])

        return right / float(batch_size)

    def evaluate_c(self, x, y) -> float:
        batch_size = len(x)
        results = self.feed_forward_c(x)
        right = sum([1 if results[i] == np.argmax(y[i]) else 0 for i in range(batch_size)])

        return right / float(batch_size)

    def inti_compress(self, w: List[Tensor], b: List[Tensor]) -> None:
        w1 = [homogenize_matrix(m) for m in w]
        b1 = [homogenize_translation_vector(v) for v in b]

        s = tf.eye(self.inputs + 1)

        for i in range(len(w1)):
            s = tf.nn.sigmoid(tf.matmul(tf.matmul(s, w1[i]), b1[i]))

        self.m = tf.Variable(s)
        self.compress = tf.assign(self.m, s)

        x = homogenize_vector(self.x)
        print(self.x.shape.as_list())
        print(x.shape.as_list())

        y = tf.matmul(x, self.m)

        yhat = tf.slice(y, [0, 0], [1, self.outputs])
        print(yhat.get_shape())
        self.predict_c = tf.argmax(yhat, axis=0)

    def __del__(self) -> None:
        self.sess.close()
