
import numpy as np

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

        if iteration % (int(num_iter / 10)) == 0:
            print(f'loss: {loss(h, y)}')

    return theta

def predict(X, theta, threshold=0.5):
    X = add_ones_feature(X)
    return sigmoid(np.dot(X, theta)) >= threshold


def rbf_kernel(v1, v2, sigma=1.0):
    return np.exp((-1 * np.linalg.norm(v1 - v2) ** 2) / (2 * sigma ** 2))

def compute_kmat(X):
    m = X.shape[0]
    kmat = np.zeros((m, m))
    for i in range(m):
        for j in range(int(m/2) + 1):
            kmat[i, j] = kmat[j, i] = rbf_kernel(X[i], X[j])
    return kmat

def compute_kmat2(X, train):
    m = X.shape[0]
    d = train.shape[0]
    kmat = np.zeros((m, d))
    for i in range(m):
        for j in range(d):
            kmat[i, j] = rbf_kernel(X[i], train[j])
    return kmat

def k_fit(X, y, learn_rate=0.01, num_iter=100000):
    kmat = compute_kmat(X)
    kmat = add_ones_feature(kmat)
    m = kmat.shape[0]
    d = kmat.shape[1]
    theta = np.zeros(d)

    for iteration in range(num_iter):
        h = sigmoid(np.dot(kmat, theta))
        gradient = np.dot(kmat.T, (h - y)) / m
        theta -= learn_rate * gradient

        if iteration % (int(num_iter / 10)) == 0:
            print(f'loss: {loss(h, y)}')

    return theta

def k_predict(X, train, theta, threshold=0.5):
    kvec = compute_kmat2(X, train)
    kvec = add_ones_feature(kvec)
    return sigmoid(np.dot(kvec, theta)) >= threshold

if __name__ == '__main__':
    data = np.genfromtxt('data2.data', delimiter=' ')
    X = data[:,:2]
    y = data[:,2]

    theta = fit(X, y)
    p = predict(X, theta)
    print(f'accuracy: {int((p == y).mean() * 100)}%')

    theta = k_fit(X, y)
    p = k_predict(X, X, theta)
    print(f'accuracy: {int((p == y).mean() * 100)}%')