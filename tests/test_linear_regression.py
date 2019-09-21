from linear_regression import LinearRegression
import numpy as np
import pytest

def test_linear_regression():
    X = np.random.normal(size=(100, 2))
    y = X[:, 0] * 5
    lr = LinearRegression()
    lr.fit(X, y)
    assert pytest.approx(lr.predict(X)[-1], rel=1e-3) == y[-1]
