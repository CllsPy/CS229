import numpy as np
import pandas as pd

def gradient_descent(func, grad, x_start, learning_rate, iterations):
    """
    Perform gradient descent optimization.
    
    Parameters:
    - func: The function to minimize.
    - grad: The gradient of the function.
    - x_start: Initial value of x.
    - learning_rate: The learning rate (step size).
    - iterations: Number of iterations to perform.
    
    Returns:
    - history: A Pandas DataFrame containing x and f(x) values at each step.
    """
    x = x_start
    x_history = [x]
    y_history = [func(x)]

    for _ in range(iterations):
        grad_value = grad(x)
        x = x - learning_rate * grad_value
        x_history.append(x)
        y_history.append(func(x))

    # Return as a DataFrame for easy display
    return pd.DataFrame({
        "Iteration": range(len(x_history)),
        "x": x_history,
        "f(x)": y_history
    })
