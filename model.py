# model.py
import numpy as np

def generate_data(n_samples=20):
    """
    Simulate housing prices based on size (X) and price (y)
    """
    np.random.seed(42)
    X = np.random.randint(50, 250, size=(n_samples, 1))  # house size in sq.m
    y = 50 + 0.3 * X + np.random.randn(n_samples, 1) * 10  # price in 1000s
    return X, y

def compute_loss(theta, X, y):
    m = len(y)
    predictions = X.dot(theta[1:]) + theta[0]
    loss = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return loss

def gradient_descent(X, y, lr=0.01, epochs=50):
    m, n = X.shape
    theta = np.random.randn(n+1, 1)  # [bias, slope]
    history = []

    for _ in range(epochs):
        predictions = X.dot(theta[1:]) + theta[0]
        error = predictions - y
        grad0 = (1/m) * np.sum(error)
        grad1 = (1/m) * X.T.dot(error)
        theta[0] -= lr * grad0
        theta[1:] -= lr * grad1
        history.append(theta.copy())
    return theta, np.array(history)
