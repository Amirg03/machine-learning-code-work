import numpy as np


class LinearRegression:
    def __init__(self, method='gradient_descent'):
        self.method = method
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if self.method == 'gradient_descent':
            return self._fit_gradient_descent(X, y, epochs, learning_rate)
        elif self.method == 'normal_equation':
            return self._fit_normal_equation(X, y)
        else:
            raise ValueError("Method must be 'gradient_descent' or 'normal_equation'")

    def _fit_gradient_descent(self, X, y, epochs, learning_rate):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

        return self

    def _fit_normal_equation(self, X, y):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        theta = np.linalg.pinv(X_bias).dot(y)

        self.bias = theta[0]
        self.weights = theta[1:]
        self.loss_history = []

        return self

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model is not fitted yet. Call fit before predict.")

        X = np.asarray(X, dtype=float)
        return np.dot(X, self.weights) + self.bias
