from utils.utils import gradient_descent
import numpy as np

function = lambda x: np.sin(x)
gradient = lambda x: np.cos(x)
learning_rate = 0.01
iterations = 10
initial_x = 0

history = gradient_descent(function, gradient, initial_x, learning_rate, iterations)
print(history)