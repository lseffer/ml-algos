from logistic_regression import LogisticRegression
import numpy as np
import pytest

def test_logistic_regression():
    X = np.random.normal(size=(100, 2))
    y = np.where(X[:, 0] > 0.5, 1, 0).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(X, y)
    pred = 1 if lr.predict(X)[-1] > 0.5 else 0
    assert pytest.approx(pred) == y[-1]
