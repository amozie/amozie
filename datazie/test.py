import numpy as np
import datazie as dz


def test1():
    x = np.linspace(0, 10, 50)
    y = 3*x + 2
    y += np.random.randn(50)
    lm = dz.model.LinearModel(y, x)
    lm.fit()
    print(lm.predict(50))

if __name__ == '__main__':
    test1()