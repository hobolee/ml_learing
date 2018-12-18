#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np


# data = pd.read_csv('Advertising.csv')
# print(data.head())
# print(type(data))
# print(data.shape)

# if __name__ == "__main__":
#     data_path = 'Advertising.csv'
#     data = pd.read_csv(data_path)
#     x = data[['TV', 'radio', 'newspaper']]
#     y = data['sales']
#     X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
#     print(X_train.shape, y_train.shape)
#     model = LinearRegression().fit(X_train, y_train)
#     print(model)
#     print(model.coef_)
#     print(model.intercept_)
#     pred = model.predict(X_test)
#     sum_mean = 0
#     for i in range(len(pred)):
#         sum_mean += (pred[i] - y_test.values[i]) ** 2
#     print("RMSE by hand:", np.sqrt(sum_mean / len(pred)))
#
#     plt.figure()
#     plt.plot(range(50), y_test, c='r')
#     plt.plot(range(50), pred.reshape([50]), c='b')
#     plt.show()


data_path = 'Advertising.csv'
data = pd.read_csv(data_path)
x = data[['TV', 'radio', 'newspaper']]
y = data['sales']
y = y.values.reshape([200, 1])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=2)
print(X_train.shape)

xs = tf.placeholder(tf.float32, [None, 3])
ys = tf.placeholder(tf.float32, [None, 1])
def add_layer(inputs, in_size, out_size, activation_function = None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    W_plus_b = tf.matmul(inputs, weights)+biases
    if activation_function is None:
        outputs = W_plus_b
    else:
        outputs = activation_function(W_plus_b)
    return outputs

h1 = add_layer(xs, 3, 2, activation_function=tf.nn.relu)
# h2 = add_layer(h1, 16, 32, activation_function=tf.nn.relu)
# h3 = add_layer(h2, 128, 32, activation_function=tf.nn.relu)
prediction = add_layer(h1, 2, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
isTrain = 0
train_steps = 100000
checkpoint_steps = 100
checkpoint_dir = 'save/'
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if isTrain:
        for i in range(train_steps):
            sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
            if (i+1) % checkpoint_steps == 0:
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)
                print(sess.run(loss, feed_dict={xs: X_test, ys: y_test}))
    else:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
    prediction = sess.run(prediction, feed_dict={xs: X_test})
print(x['TV'].shape, prediction.shape)
plt.figure()
plt.plot(range(len(X_test)), y_test, c='r')
plt.plot(range(len(X_test)), prediction, c='b')
plt.show()

