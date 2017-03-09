import matplotlib.pyplot as plt
import numpy as np
import random

from load_data import load_data
from losses import logistic_loss, logistic_loss_gradient

# SGD: No averaging vs. averaging
def plot_1(Xtrain, ytrain, Xtest, ytest):
    step = 0.5
    iterations = 20

    f = logistic_loss
    gradf = logistic_loss_gradient

    n, d = Xtrain.shape

    theta = np.zeros((d, 1))
    theta_average = np.zeros((d, 1))
    loss_test = [f(Xtest, ytest, theta_average)]
    loss_test_average = [f(Xtest, ytest, theta_average)]

    for it in range(iterations):
        for i in range(n):
            g = gradf(Xtrain[i, :][np.newaxis, :], ytrain[i], theta)
            m = it * n + i + 1
            theta -= step * g / np.sqrt(m)
            theta_average = ((m - 1) * theta_average + theta) / m

            if m % 100 == 0:
                loss_test.append(f(Xtest, ytest, theta))
                loss_test_average.append(f(Xtest, ytest, theta_average))

    interval = range(len(loss_test))
    interval = [100 * x for x in interval]

    plt.figure(1)
    plt.plot(interval, loss_test, 'b', interval, loss_test_average, 'r')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend(['normal', 'averaged'], loc='upper left')
    plt.show()

# Strongly-convex vs. convex (averaged SGD)
def plot_2(Xtrain, ytrain, Xtest, ytest, mu):
    step = 0.1
    iterations = 1

    f = logistic_loss
    gradf = logistic_loss_gradient

    n, d = Xtrain.shape

    theta = np.zeros((d, 1))
    theta_average = np.zeros((d, 1))
    loss_test = [f(Xtest, ytest, theta_average)]

    for it in range(iterations):
        for i in range(n):
            g = gradf(Xtrain[i, :][np.newaxis, :], ytrain[i], theta)
            m = it * n + i + 1
            theta -= step * g / np.sqrt(m)
            theta_average = ((m - 1) * theta_average + theta) / m

            if m % 100 == 0:
                loss_test.append(f(Xtest, ytest, theta_average))

    theta = np.zeros((d, 1))
    theta_average = np.zeros((d, 1))
    loss_test_mu = [f(Xtest, ytest, theta_average)]

    for it in range(iterations):
        for i in range(n):
            g = gradf(Xtrain[i, :][np.newaxis, :], ytrain[i], theta, mu=mu)
            m = it * n + i + 1
            theta -= step * g / np.sqrt(m)
            theta_average = ((m - 1) * theta_average + theta) / m

            if m % 100 == 0:
                loss_test_mu.append(f(Xtest, ytest, theta_average))

    interval = range(len(loss_test))
    interval = [100 * x for x in interval]

    plt.figure(1)
    plt.plot(interval, loss_test, 'b', interval, loss_test_mu, 'r')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend(['convex', 'strongly convex'], loc='upper left')
    plt.show()

# SGD: cycling vs. sampling with replacement
def plot_3(Xtrain, ytrain, Xtest, ytest):
    step = 0.5
    iterations = 20

    f = logistic_loss
    gradf = logistic_loss_gradient

    n, d = Xtrain.shape

    theta = np.zeros((d, 1))
    loss_test = [f(Xtest, ytest, theta)]

    for it in range(iterations):
        for i in range(n):
            choose = random.randint(0,n - 1)
            g = gradf(Xtrain[i, :][np.newaxis, :], ytrain[i], theta)
            m = it * n + i + 1
            theta -= step * g / np.sqrt(m)

            if m % 100 == 0:
                loss_test.append(f(Xtest, ytest, theta))

    theta = np.zeros((d, 1))
    loss_test_sampling = [f(Xtest, ytest, theta)]

    for it in range(iterations):
        for i in range(n):
            choose = random.randint(0,n - 1)
            g = gradf(Xtrain[choose, :][np.newaxis, :], ytrain[choose], theta)
            m = it * n + i + 1
            theta -= step * g / np.sqrt(m)

            if m % 100 == 0:
                loss_test_sampling.append(f(Xtest, ytest, theta))

    interval = range(len(loss_test))
    interval = [100 * x for x in interval]

    plt.figure(1)
    plt.plot(interval, loss_test, 'b', interval, loss_test_sampling, 'r')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend(['cycling', 'sampling'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    Xtrain, ytrain, Xtest, ytest = load_data()
    plot_1(Xtrain, ytrain, Xtest, ytest)
    plot_2(Xtrain, ytrain, Xtest, ytest, 0.1)
    plot_3(Xtrain, ytrain, Xtest, ytest)
    plt.show()
