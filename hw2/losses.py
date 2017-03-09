import numpy as np

def logistic_loss(X, y, theta):
    n = X.shape[0]
    aux = -y * np.dot(X, theta)
    return 1 / float(n) * np.sum(np.log(1 + np.exp(aux)))

def logistic_loss_gradient(X, y, theta, mu=0):
    n, d = X.shape
    g = np.zeros((d, 1))
    aux = -y * np.dot(X, theta)

    for i in range(n):
        g += 1.0 / (1 + np.exp(-aux[i])) * -y[i] * np.expand_dims(X[i,:].T, axis=1)
    g /= float(n)
    g += mu * theta

    return g
