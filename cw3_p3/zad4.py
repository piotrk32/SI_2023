import numpy as np

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Set the learning rate
alpha = 0.1

# Define the input dataset and output values
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the neural network architecture
input_layer = X.shape[1]
hidden_layer = 2
output_layer = 1

# Initialize the weight matrices with random values
w1 = np.random.randn(input_layer, hidden_layer)
w2 = np.random.randn(hidden_layer, output_layer)

# Train the network
for epoch in range(10000):
    # Forward pass
    z2 = X.dot(w1)
    a2 = sigmoid(z2)
    z3 = a2.dot(w2)
    y_pred = sigmoid(z3)

    # Backward pass
    delta3 = (y_pred - y) * sigmoid_derivative(y_pred)
    delta2 = delta3.dot(w2.T) * sigmoid_derivative(a2)

    # Update weights
    w2 -= alpha * a2.T.dot(delta3)
    w1 -= alpha * X.T.dot(delta2)

# Print the final predictions
print(y_pred)