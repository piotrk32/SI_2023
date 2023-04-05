import numpy as np


#Sieć składa się z trzech warstw: warstwy wejściowej, warstwy ukrytej i warstwy wyjściowej. Warstwa wejściowa ma dwa neurony,
# jeden dla każdego z dwóch wejść. Warstwa ukryta ma trzy neurony,
# a warstwa wyjściowa ma jeden neuron. Wszystkie neurony w warstwach ukrytej i wyjściowej korzystają z funkcji aktywacji sigmoidalnej.


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, num_hidden, alpha):
    np.random.seed(42)
    n, m = X.shape
    w = np.random.rand(m, num_hidden)
    w0 = np.random.rand(1, num_hidden)
    v = np.random.rand(num_hidden, 1)
    v0 = np.random.rand(1, 1)

    for i in range(10000):
        # Forward propagation
        z = sigmoid(X @ w + w0)
        a = sigmoid(z @ v + v0)

        # Backward propagation
        error = y - a
        delta_a = error * a * (1 - a)
        delta_z = delta_a @ v.T * z * (1 - z)
        v += alpha * z.T @ delta_a
        v0 += alpha * np.sum(delta_a, axis=0, keepdims=True)
        w += alpha * X.T @ delta_z
        w0 += alpha * np.sum(delta_z, axis=0)

    return w, w0

X = np.array([[0.6, 0.1], [0.2, 0.3]])
y = np.array([[1], [0]])

y = np.reshape(y, (-1, 1))  # Reshape y to (n, 1)

w, w0 = train(X, y, 4, 0.5)

print("Weights:")
print("w = ", w)
print("w0 = ", w0)
