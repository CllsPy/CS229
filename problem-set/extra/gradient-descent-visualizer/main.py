import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import gradient_descent

# Title
st.title("Gradient Descent Visualizer")

# Sidebar Controls
st.sidebar.header("Parameters")
learning_rate = st.sidebar.slider("Learning Rate (α)", min_value=0.001, max_value=1.0, value=0.1, step=0.001)
initial_x = st.sidebar.slider("Initial x", min_value=-10.0, max_value=10.0, value=5.0, step=0.1)
iterations = st.sidebar.slider("Iterations", min_value=1, max_value=100, value=20, step=1)

# Function Definition
st.sidebar.header("Function Settings")
function_choice = st.sidebar.selectbox(
    "Choose the function to optimize:",
    ["x^2", "2x^2 + 2", "sin(x)"]
)


madeBy = st.sidebar.write("made with <3")

if function_choice == "x^2":
    function = lambda x: x**2
    gradient = lambda x: 2*x
elif function_choice == "2x^2 + 2":
    function = lambda x: 2*x**2 + 2
    gradient = lambda x: 4*x 
else:
    function = lambda x: np.sin(x)
    gradient = lambda x: np.cos(x)

# Gradient Descent Computation
history = gradient_descent(function, gradient, initial_x, learning_rate, iterations)

# Plotting
x_vals = np.linspace(-10, 10, 500)
y_vals = function(x_vals)

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label=f"f(x) = {function_choice}")
ax.scatter(history['x'], history['f(x)'], color='red', label="Gradient Descent Steps")
ax.plot(history['x'], history['f(x)'], linestyle='--', color='red', alpha=0.6)
ax.set_title("Gradient Descent Visualization")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()

st.pyplot(fig)

# with col2: 
    # Display Data
    #st.subheader("Gradient Descent Steps")
    #st.write("Initial x:", initial_x)
    #st.write("Learning Rate:", learning_rate)
    #st.write("Number of Iterations:", iterations)

#    with st.expander("See explanation"):
#        st.table(history)
