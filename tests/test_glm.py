from glm import PoissonGLM
import numpy as np
import pytest

def test_poisson_glm():
    X = np.random.normal(size=(100, 10))
    y = np.random.poisson(10, size=(100,))
    gm = PoissonGLM()
    gm.fit(X, y)
    assert pytest.approx(np.average(gm.predict(X)), abs=1) == 10
