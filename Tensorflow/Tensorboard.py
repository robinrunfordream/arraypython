#tensorboard 檔案無法生成 待改
from __future__ import print_function
import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random.normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs

# 關閉 eager execution
tf.compat.v1.disable_eager_execution()

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.compat.v1.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        axis=[1]))

with tf.name_scope('train'):
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化變數
init = tf.compat.v1.global_variables_initializer()

# 啟動 session
with tf.compat.v1.Session() as sess:
    # 寫入 TensorBoard 的 log 檔案
    writer = tf.summary.create_file_writer("test/")
    tf.summary.trace_on(graph=True, profiler=True)

    # 初始化變數
    sess.run(init)

    # ... 其他訓練程式碼
