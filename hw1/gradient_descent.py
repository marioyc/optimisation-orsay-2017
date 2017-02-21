import matplotlib.pyplot as plt
import numpy as np

from load_data import load_data
from losses import (square_loss, square_loss_gradient,
                logistic_loss, logistic_loss_gradient)

def classification_error(y, y2):
    n = y.shape[0]
    error = 0
    aux = y * y2
    for i in range(n):
        if aux[i] < 0 or (aux[i] == 0 and y[i] == -1):
            error += 1
    return error / float(n) * 100

if __name__ == '__main__':
    step = 0.5
    loss = 'logistic'
    iterations = 500

    if loss == 'square':
        f = square_loss
        gradf = square_loss_gradient
    elif loss == 'logistic':
        f = logistic_loss
        gradf = logistic_loss_gradient
    else:
        raise Exception("Invalid loss")

    Xtrain, ytrain, Xtest, ytest = load_data()
    d = Xtrain.shape[1]
    theta = np.zeros((d,1))

    loss_train = [f(Xtrain, ytrain, theta)]
    loss_test = [f(Xtest, ytest, theta)]
    error_train = [classification_error(ytrain, np.dot(Xtrain, theta))]
    error_test = [classification_error(ytest, np.dot(Xtest, theta))]

    for it in range(iterations):
        g = gradf(Xtrain, ytrain, theta)
        theta -= step * g
        loss_train.append(f(Xtrain, ytrain, theta))
        loss_test.append(f(Xtest, ytest, theta))
        error_train.append(classification_error(ytrain, np.dot(Xtrain, theta)))
        error_test.append(classification_error(ytest, np.dot(Xtest, theta)))

    plt.figure(1)
    plt.plot(range(iterations + 1), loss_train, 'b', range(iterations + 1), loss_test, 'r')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.figure(2)
    plt.plot(range(iterations + 1), error_train, 'b', range(iterations + 1), error_test, 'r')
    plt.ylabel('classification error (%)')
    plt.xlabel('iteration')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()
