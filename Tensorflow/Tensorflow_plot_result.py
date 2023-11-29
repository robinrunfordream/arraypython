from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define variables for inputs to network
xs = tf.Variable(initial_value=x_data, dtype=tf.float32)
ys = tf.Variable(initial_value=y_data, dtype=tf.float32)
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data
def compute_loss():
    return tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), axis=[1]))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
# `minimize` will compute the gradients and apply them automatically
train_step = optimizer.minimize(compute_loss, var_list=[xs, ys])

# 不再需要初始化
# init = tf.compat.v1.global_variables_initializer()
# sess = tf.compat.v1.Session()
# sess.run(init)
# 不再需要 Session 的開始和關閉
# for i in range(1000):
#     # training
#     sess.run(train_step)
#     if i % 50 == 0:
#         # to see the step improvement
#         print(sess.run(loss()))

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()



for i in range(1000):
    # training
    optimizer.minimize(compute_loss, var_list=[xs, ys])
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        # 原本用SESS.run但是舊版才可以
        # prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # # plot the prediction
        # lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        # plt.pause(0.1)

   
  
        xs.assign(x_data)
        # Use the model to predict
        prediction_value = prediction.numpy() #prediction_value 是錯誤的
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
