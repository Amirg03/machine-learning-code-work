import numpy as np


class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _binary_cross_entropy(self, y, y_prob):
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        return -np.mean(y * np.log(y_prob) + (1.0 - y) * np.log(1.0 - y_prob))

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(epochs):
            linear_output = np.dot(X, self.weights) + self.bias
            y_prob = self._sigmoid(linear_output)

            loss = self._binary_cross_entropy(y, y_prob)
            self.loss_history.append(float(loss))

            error = y_prob - y
            dw = np.dot(X.T, error) / n_samples
            db = np.mean(error)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

        return self

    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X):
        y_prob = self.predict_proba(X)
        return (y_prob >= 0.5).astype(int)
