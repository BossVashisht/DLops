import numpy as np
import matplotlib.pyplot as plt

# Define the random values
randomValues = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

plt.tight_layout()
plt.show()