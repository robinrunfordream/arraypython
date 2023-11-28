import tensorflow as tf

state = tf.Variable(0, name='counter')
# print(state.name)

# 創建一個 TensorFlow 常數 one，值為 1
one = tf.constant(0)

# 使用 tf.add 函數將 state 和 one 相加，得到新的值
new_value = tf.add(state, one)
state.assign(new_value)  # 直接使用 tf.Variable 的 assign 方法


# 使用 for 迴圈執行 3 次  
for _ in range(3):
    state.assign_add(1)  # 使用 assign_add 方法，每次增加 1
    print(state.numpy())  # 在 eager execution 中，您可以直接使用 `.numpy()` 獲取變數的值
