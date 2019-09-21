import numpy as np
from typing import Tuple, List, Union

class LogisticRegression():

    def __init__(self, iterations=1000, learning_rate=1e-2):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def sigmoid(self, x: Union[float, np.ndarray], derivative=False) -> Union[float, np.ndarray]:
        sigm = 1. / (1. + np.exp(-x))
        if derivative:
            return sigm * (1. - sigm)
        return sigm

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float:
        n = y_true.shape[0]
        return 1 / n * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).sum()

    def loss_derivative(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.float]:
        return ((y_pred - y_true).reshape(-1, 1) * X).sum(axis=0), (y_pred - y_true).reshape(-1, 1).sum(axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        assert len(y.shape) == 2, "y has to have 2 dimensions"
        self.thetas = np.random.normal(size=(X.shape[1] + 1, ))
        self.losses: List = []
        for iteration in range(self.iterations):
            y_pred = self.predict(X)
            self.losses.append(self.loss(y, y_pred))
            loss_derivative = self.loss_derivative(X, y, y_pred)
            self.thetas[1:] = self.thetas[1:] - self.learning_rate * loss_derivative[0]
            self.thetas[:1] = self.thetas[:1] - self.learning_rate * loss_derivative[1]

    def predict(self, X):
        return self.sigmoid(np.dot(self.thetas[1:], X.T) + self.thetas[:1]).reshape(-1, 1)
