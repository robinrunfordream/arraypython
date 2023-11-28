from __future__ import print_function
import tensorflow as tf
import numpy as np

#這個程式碼是使用 TensorFlow 2.x 的 Eager Execution 模式來實現一個簡單的線性回歸（linear regression）模型

# create data
x_data = np.random.rand(100).astype(np.float32)#是一個包含 100 個浮點數的陣列，每個數字都在 0 到 1 之間。
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))  # 是一個可訓練的變數（tf.Variable），表示模型的權重，初始值在 -1 到 1 之間隨機均勻分佈
biases = tf.Variable(tf.zeros([1]))#同樣是一個可訓練的變數，表示模型的偏差，初始值為 0

def compute_loss(): #數計算模型的均方誤差（Mean Squared Error）損失，即模型預測的 y 與真實 y 之間的平方差的平均值
    y = Weights * x_data + biases
    return tf.reduce_mean(tf.square(y - y_data))

# 使用 tf.keras.optimizers.SGD 替代 GradientDescentOptimizer
optimizer = tf.keras.optimizers.SGD(0.5) #SGD（Stochastic Gradient Descent）優化器，這是一種基本的優化器，用於最小化損失函數

""" 訓練過程：@tf.function 裝飾器將 train_step 函數轉換為 TensorFlow 的計算圖，這樣可以提高計算性能。
在每個訓練步驟中，使用 tf.GradientTape 記錄計算梯度的操作。這是為了後續的梯度下降更新權重。
optimizer.apply_gradients 將計算得到的梯度應用到權重和偏差上，實現梯度下降的一步。
訓練過程中，模型的權重和偏差會不斷調整，以使預測的 y 接近實際的 y_data。
結果顯示：

在每 20 步，印出當前的權重和偏差。
總的來說，這個程式碼演示了如何使用 TensorFlow 2.x 來建立一個簡單的線性回歸模型，並使用梯度下降法來最小化預測值和實際值之間的均方誤差。 """


@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss = compute_loss()
    gradients = tape.gradient(loss, [Weights, biases])
    optimizer.apply_gradients(zip(gradients, [Weights, biases]))

### create tensorflow structure end ###

# 不再需要 Session()，也不需要初始化
# sess = tf.compat.v1.Session()

for step in range(501):
    train_step()
    if step % 20 == 0:
        print(step, Weights.numpy(), biases.numpy())



