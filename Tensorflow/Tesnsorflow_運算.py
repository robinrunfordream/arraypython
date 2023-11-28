
from __future__ import print_function
import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)



# 直接執行 TensorFlow 運算
result = product.numpy()
print(result)





