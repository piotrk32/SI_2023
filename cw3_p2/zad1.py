import numpy as np

# Definicja funkcji AND
def and_func(x):
    return int(x[0] and x[1])

# Definicja funkcji NOT
def not_func(x):
    return int(not x[0])

# Definicja funkcji aktywacji (step function)
def step_func(x):
    return np.where(x >= 0, 1, 0)

# Perceptron Learn Algorithm
def perceptron_learn(x, y, w, b, learning_rate):
    epochs = 100
    for epoch in range(epochs):
        errors = 0
        for i in range(len(x)):
            # Obliczenie net
            net = np.dot(x[i], w) + b
            # Obliczenie y_hat
            y_hat = step_func(net)
            # Obliczenie błędu
            error = y[i] - y_hat
            if error != 0:
                errors += 1
                # Aktualizacja w i b
                w = w + learning_rate * error * x[i]
                b = b + learning_rate * error
        if errors == 0:
            print("Training converged at epoch", epoch+1)
            break
    return w, b

# Ustawienie danych treningowych i oczekiwanych wyników
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_not = np.array([1, 0, 1, 0])

# Inicjalizacja wag i biasu
w_and = np.zeros(2)
b_and = 0
w_not = np.zeros(1)
b_not = 0

# Uczenie perceptronów dla funkcji AND i NOT
learning_rate = 0.1
w_and, b_and = perceptron_learn(x_train, y_and, w_and, b_and, learning_rate)
w_not, b_not = perceptron_learn(x_train[:, [0]], y_not, w_not, b_not, learning_rate)

# Testowanie perceptronów
for x in x_train:
    print("AND({}, {}) = {}".format(x[0], x[1], and_func(x)))
    print("NOT({}) = {}".format(x[0], not_func(x)))
    net_and = np.dot(x, w_and) + b_and
    y_hat_and = step_func(net_and)
    net_not = np.dot(x.reshape(-1, 1), w_not) + b_not
    y_hat_not = step_func(net_not)
    print("AND({}, {}) = {}".format(x[0], x[1], y_hat_and))
    print("NOT({}) = {}".format(x[0], y_hat_not))