import numpy as np
import matplotlib.pyplot as plt




# 定義激勵函數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# 生成 x 值
x = np.linspace(-5, 5, 100)

# 計算激勵函數的輸出
sigmoid_output = sigmoid(x)
relu_output = relu(x)
tanh_output = tanh(x)

# 繪製圖表
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(x, sigmoid_output, label='Sigmoid')
plt.title('Sigmoid')
plt.legend()

plt.subplot(132)
plt.plot(x, relu_output, label='ReLU')
plt.title('ReLU')
plt.legend()

plt.subplot(133)
plt.plot(x, tanh_output, label='Tanh')
plt.title('Tanh')
plt.legend()

plt.tight_layout()
plt.show()
