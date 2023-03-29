import numpy as np

# Definicja funkcji boolowskiej x1 ∧ ¬x2
def bool_func(x):
    return int(x[0] and not x[1])

# Definicja funkcji aktywacji (step function)
def step_func(x):
    return np.where(x >= 0, 1, 0)

# Perceptron z dwoma wejściami reprezentujący funkcję boolowską x1 ∧ ¬x2
def perceptron(x, w, b):
    net = np.dot(x, w) + b
    y_hat = step_func(net)
    return y_hat

# Ustawienie danych treningowych i oczekiwanych wyników
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 1, 0])

# Inicjalizacja wag i biasu
w = np.array([1, -1, -1])
b = 0

# Uczenie perceptronu
learning_rate = 0.1
epochs = 100
for epoch in range(epochs):
    errors = 0
    for i in range(len(x_train)):
        x = np.array([x_train[i][0], not x_train[i][1], 1])
        y = y_train[i]
        y_hat = perceptron(x, w, b)
        error = y - y_hat
        if error != 0:
            errors += 1
            w = w + learning_rate * error * x
            b = b + learning_rate * error
    if errors == 0:
        print("Training converged at epoch", epoch+1)
        break

# Testowanie perceptronu
for x in x_train:
    y_hat = perceptron(np.array([x[0], not x[1], 1]), w, b)
    print("x1 = {}, x2 = {}, x1 ∧ ¬x2 = {}".format(x[0], x[1], bool_func(x)))
    print("Perceptron output: {}".format(y_hat))
