import numpy as np

# Definicja funkcji boolowskiej XOR
def bool_func(x):
    return int(x[0] != x[1])

# Definicja funkcji aktywacji (step function)
def step_func(x):
    return np.where(x >= 0, 1, 0)

# Dwuwarstwowa sieć perceptronów implementująca funkcję boolowską XOR
def neural_network(x, w1, w2, b1, b2):
    # Propagacja w przód - pierwsza warstwa perceptronów
    net1 = np.dot(x, w1) + b1
    y1 = step_func(net1)
    # Propagacja w przód - druga warstwa perceptronów
    net2 = np.dot(y1, w2) + b2
    y_hat = step_func(net2)
    return y_hat

# Ustawienie danych treningowych i oczekiwanych wyników
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])

# Inicjalizacja wag i biasów
w1 = np.random.rand(2, 2)
w2 = np.random.rand(2)
b1 = np.zeros(2)
b2 = 0

# Uczenie sieci perceptronów
learning_rate = 0.1
epochs = 1000
for epoch in range(epochs):
    errors = 0
    for i in range(len(x_train)):
        x = x_train[i]
        y = y_train[i]
        # Propagacja w przód - pierwsza warstwa perceptronów
        net1 = np.dot(x, w1) + b1
        y1 = step_func(net1)
        # Propagacja w przód - druga warstwa perceptronów
        net2 = np.dot(y1, w2) + b2
        y_hat = step_func(net2)
        error = y - y_hat
        if error != 0:
            errors += 1
            # Propagacja wsteczna - druga warstwa perceptronów
            delta2 = error * y_hat * (1 - y_hat)
            w2 = w2 + learning_rate * delta2 * y_hat
            b2 = b2 + learning_rate * delta2
            # Propagacja wsteczna - pierwsza warstwa perceptronów
            delta1 = delta2 * w2 * y1 * (1 - y1)
            w1 = w1 + learning_rate * delta1.reshape(-1, 1) * x.reshape(1, -1)
            b1 = b1 + learning_rate * delta1
    if errors == 0:
        print("Training converged at epoch", epoch+1)
        break

    # Testowanie sieci perceptronów
for x in x_train:
    # Propagacja w przód - pierwsza warstwa perceptronów
    net1 = np.dot(x, w1) + b1
    y1 = step_func(net1)
    # Propagacja w przód - druga warstwa perceptronów
    net2 = np.dot(y1, w2) + b2
    y_hat = step_func(net2)
    print("x1 = {}, x2 = {}, x1 XOR x2 = {}".format(x[0], x[1], bool_func(x)))
    print("Neural network output: {}".format(y_hat))