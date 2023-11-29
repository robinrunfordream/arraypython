from __future__ import print_function
import tensorflow as tf
import numpy as np

# 創建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(2)
])

# 使用 tf.keras.losses.sparse_categorical_crossentropy
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 定義優化器
optimizer = tf.optimizers.SGD(learning_rate=0.1)

# 準備數據
x_data = np.random.rand(100, 10).astype(np.float32)
y_data = np.random.randint(0, 2, size=(100,)).astype(np.int64)

# 開始訓練
for step in range(1000):
    with tf.GradientTape() as tape:
        # 計算模型的輸出
        predictions = model(x_data)
        # 計算損失
        loss_value = loss_object(y_data, predictions)
    
    # 取得梯度
    gradients = tape.gradient(loss_value, model.trainable_variables)
    
    # 更新權重
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss_value.numpy()}")
