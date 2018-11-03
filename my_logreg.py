
import numpy as np
from time import time

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def add_ones_feature(X):
    _X = np.ones((X.shape[0], X.shape[1] + 1))
    _X[:, 1:] = X
    return _X

def fit(X, y, learn_rate=0.01, num_iter=100000):
    X = add_ones_feature(X)
    m = X.shape[0]
    d = X.shape[1]
    theta = np.zeros(d)

    for iteration in range(num_iter):
        h = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learn_rate * gradient

        if iteration % (num_iter // 10) == 0:
            print(f'loss: {loss(h, y)}')

    return theta

def fit_stochastic(X, y, learn_rate=0.01, num_iter=100000):
    X = add_ones_feature(X)
    m = X.shape[0]
    d = X.shape[1]
    theta = np.zeros(d)

    num_iter = num_iter // m
    for iteration in range(num_iter):
        for i in range(m):
            h = sigmoid(np.dot(X, theta))
            gradient = np.dot(X[i].T, (h[i] - y[i])) / m
            theta -= learn_rate * gradient

        if iteration % (num_iter // 10) == 0:
            print(f'loss: {loss(h, y)}')

    return theta


def predict(X, theta, threshold=0.5):
    X = add_ones_feature(X)
    return sigmoid(np.dot(X, theta)) >= threshold


def rbf_kernel(v1, v2, sigma=1.0):
    return np.exp((-1 * np.linalg.norm(v1 - v2) ** 2) / (2 * sigma ** 2))

def polynomial_kernel(v1, v2, c=1, d=2):
    return (v1.T.dot(v2) + c) ** d

def sigmoid_kernel(v1, v2, alpha=2, c=1):
    return np.tanh(alpha * v1.T.dot(v2) + c)

def compute_kmat(X, kernel):
    m = X.shape[0]
    kmat = np.zeros((m, m))
    for i in range(m):
        for j in range(m//2 + 1):
            kmat[i, j] = kmat[j, i] = kernel(X[i], X[j])
    return kmat

def compute_kmat2(X, train, kernel):
    m = X.shape[0]
    d = train.shape[0]
    kmat = np.zeros((m, d))
    for i in range(m):
        for j in range(d):
            kmat[i, j] = kernel(X[i], train[j])
    return kmat

def k_fit(X, y, learn_rate=0.01, num_iter=100000, kernel=rbf_kernel):
    kmat = compute_kmat(X, kernel)
    kmat = add_ones_feature(kmat)
    m = kmat.shape[0]
    d = kmat.shape[1]
    theta = np.zeros(d)

    for iteration in range(num_iter):
        h = sigmoid(np.dot(kmat, theta))
        gradient = np.dot(kmat.T, (h - y)) / m
        theta -= learn_rate * gradient

        if iteration % (num_iter // 10) == 0:
            print(f'loss: {loss(h, y)}')

    return theta

def k_fit_stochastic(X, y, learn_rate=0.01, num_iter=100000, kernel=rbf_kernel):
    kmat = compute_kmat(X, kernel)
    kmat = add_ones_feature(kmat)
    m = kmat.shape[0]
    d = kmat.shape[1]
    theta = np.zeros(d)
    num_iter = num_iter // m # performance enhancement
    for iteration in range(num_iter):
        for i in range(m):
            h = sigmoid(np.dot(kmat, theta))
            gradient = np.dot(kmat[i].T, (h[i] - y[i]))
            theta -= learn_rate * gradient

        if iteration % (num_iter // 10) == 0:
            print(f'loss: {loss(h, y)}')

    return theta

def k_predict(X, train, theta, threshold=0.5, kernel=rbf_kernel):
    kmat = compute_kmat2(X, train, kernel)
    kmat = add_ones_feature(kmat)
    return sigmoid(np.dot(kmat, theta)) >= threshold


if __name__ == '__main__':
    data = np.genfromtxt('data2.data', delimiter=' ')
    np.random.shuffle(data)
    train_size = len(data) * 60 // 100
    X_train = data[:train_size,:2]
    y_train = data[:train_size,2]
    X_test = data[train_size:,:2]
    y_test = data[train_size:,2]

    print("logistic regression")
    start = time()
    theta = fit(X_train, y_train)
    p = predict(X_test, theta)
    print(f'accuracy: {int((p == y_test).mean() * 100)}%')
    print(f'execution time: {time() - start} sec')

    print("\nkernelized logistic regression")
    start = time()
    theta = k_fit(X_train, y_train)
    p = k_predict(X_test, X_train, theta)
    print(f'accuracy: {int((p == y_test).mean() * 100)}%')
    print(f'execution time: {time() - start} sec')

    print("\nlogistic regression (stochastic GD)")
    start = time()
    theta = fit_stochastic(X_train, y_train)
    p = predict(X_test, theta)
    print(f'accuracy: {int((p == y_test).mean() * 100)}%')
    print(f'execution time: {time() - start} sec')

    print("\nkernelized logistic regression (stochastic GD)")
    start = time()
    theta = k_fit_stochastic(X_train, y_train)
    p = k_predict(X_test, X_train, theta)
    print(f'accuracy: {int((p == y_test).mean() * 100)}%')
    print(f'execution time: {time() - start} sec')
