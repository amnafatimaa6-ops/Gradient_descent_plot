# model.py
import numpy as np

def generate_data(n_samples=100):
    """Generate simple linear data"""
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1)
    return X, y

def compute_loss(theta, X, y):
    """Compute MSE loss"""
    m = len(y)
    predictions = X.dot(theta[1:]) + theta[0]
    loss = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return loss

def gradient_descent(X, y, lr=0.1, epochs=50):
    """Perform gradient descent and return theta and history"""
    m, n = X.shape
    theta = np.random.randn(n+1, 1)  # [theta0 (bias), theta1]
    loss_history = []
    for _ in range(epochs):
        predictions = X.dot(theta[1:]) + theta[0]
        error = predictions - y
        grad0 = (1/m) * np.sum(error)
        grad1 = (1/m) * X.T.dot(error)
        theta[0] -= lr * grad0
        theta[1:] -= lr * grad1
        loss_history.append(theta.copy())
    return theta, np.array(loss_history)
