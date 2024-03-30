import numpy as np
import matplotlib.pyplot as plt

# Define the random values
randomValues = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh activation function
def tanh(x):
    return np.tanh(x)

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU activation function
def leakyRelu(x, alpha=0.1):
    return np.maximum(alpha*x, x)

# Generate x values for plotting
x = np.linspace(-5, 5, 100)

# Plotting
plt.figure(figsize=(12, 8))

# Sigmoid plot
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.scatter(randomValues, sigmoid(randomValues), color='red')
plt.title('Sigmoid Activation Function')
plt.legend()

# Tanh plot
plt.subplot(2, 2, 2)
plt.plot(x, tanh(x), label='Tanh')
plt.scatter(randomValues, tanh(randomValues), color='red')
plt.title('Tanh Activation Function')
plt.legend()

# ReLU plot
plt.subplot(2, 2, 3)
plt.plot(x, relu(x), label='ReLU')
plt.scatter(randomValues, relu(randomValues), color='red')
plt.title('ReLU Activation Function')
plt.legend()

# Leaky ReLU plot
plt.subplot(2, 2, 4)
plt.plot(x, leakyRelu(x), label='Leaky ReLU')
plt.scatter(randomValues, leakyRelu(randomValues), color='red')
plt.title('Leaky ReLU Activation Function')
plt.legend()

plt.tight_layout()
plt.show()