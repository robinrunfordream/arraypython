from __future__ import print_function
import tensorflow as tf

# 直接使用 Python 變數
input1 = tf.constant([7.], dtype=tf.float32)
input2 = tf.constant([2.], dtype=tf.float32)

# 直接進行乘法操作
output = tf.multiply(input1, input2)

# eager execution 下可以直接使用 .numpy() 取得數值
result = output.numpy()
print(result)
