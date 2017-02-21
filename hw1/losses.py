import numpy as np

def square_loss(X, y, theta):
    n = X.shape[0]
    return 1 / float(n) * np.sum((y - np.dot(X, theta)) ** 2)

def square_loss_gradient(X, y, theta):
    n, d = X.shape
    g = np.zeros((d, 1))
    aux = y - np.dot(X, theta)

    for i in range(n):
        g += aux[i] * np.expand_dims(X[i,:].T, axis=1)

    return -2 / float(n) * g

def logistic_loss(X, y, theta):
    n = X.shape[0]
    aux = -y * np.dot(X, theta)
    return 1 / float(n) * np.sum(np.log(1 + np.exp(aux)))

def logistic_loss_gradient(X, y, theta):
    n, d = X.shape
    g = np.zeros((d, 1))
    aux = -y * np.dot(X, theta)

    for i in range(n):
        g += 1.0 / (1 + np.exp(-aux[i])) * -y[i] * np.expand_dims(X[i,:].T, axis=1)

    return 1 / float(n) * g
