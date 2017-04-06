import matplotlib.pyplot as plt
import numpy as np
import random

from load_data import load_data
from losses import (square_loss, square_loss_gradient,
                logistic_loss, logistic_loss_gradient)

# SGD (sampling with replacement) vs SGA
def plot(Xtrain, ytrain, Xtest, ytest, loss_name, mu):
    step = 0.01
    iterations = 50

    if loss_name == 'square':
        f = square_loss
        gradf = square_loss_gradient
    elif loss_name == 'logistic':
        f = logistic_loss
        gradf = logistic_loss_gradient

    n, d = Xtrain.shape

    theta = np.zeros((d, 1))
    loss_test_sag = [f(Xtest, ytest, theta)]
    y = np.zeros((n, d))
    sumy = np.zeros((d, 1))

    for it in range(iterations):
        for i in range(n):
            choose = random.randint(0,n - 1)
            #print Xtrain[choose, :], ytrain[choose]
            g = gradf(Xtrain[choose, :][np.newaxis, :], ytrain[choose], theta, mu)
            sumy[:, 0] += g[:, 0] - y[choose, :]
            y[choose, :] = g[:, 0]
            #g[:, 0] -= y[choose, :]
            #print g#sumy, theta
            #print g.shape
            #print y[choose, :].shape
            #print (g - y[choose, :]).shape
            #print sumy.shape
            #print sumy
            #assert False
            theta -= step * sumy / n

            m = it * n + i + 1
            if m % 100 == 0:
                loss_test_sag.append(f(Xtest, ytest, theta))

    theta = np.zeros((d, 1))
    loss_test_sampling = [f(Xtest, ytest, theta)]

    for it in range(iterations):
        for i in range(n):
            choose = random.randint(0,n - 1)
            g = gradf(Xtrain[choose, :][np.newaxis, :], ytrain[choose], theta, mu)
            theta -= step * g#step * g / np.sqrt(m)

            m = it * n + i + 1
            if m % 100 == 0:
                loss_test_sampling.append(f(Xtest, ytest, theta))

    interval = range(len(loss_test_sag))
    interval = [100 * x for x in interval]

    plt.figure(1)
    plt.plot(interval, loss_test_sag, 'b', interval, loss_test_sampling, 'r')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend(['SAG', 'SGD with sampling'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    Xtrain, ytrain, Xtest, ytest = load_data()
    plot(Xtrain, ytrain, Xtest, ytest, 'logistic', 1e-1)
    plt.show()
