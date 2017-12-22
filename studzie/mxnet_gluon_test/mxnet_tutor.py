import mxnet.ndarray as nd
import mxnet.autograd as ag
from mxnet import gluon

nd.zeros((3, 4))
x = nd.ones((3, 4))
nd.array([[1, 2], [3, 4]])
y = nd.random.uniform(-1, 1, (3, 4))

a = nd.arange(12).reshape((3, 4))


def f(x):
    return x**3

from mxnet import ndarray as nd
from mxnet import autograd

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

import matplotlib.pyplot as plt
plt.scatter(X[:, 1].asnumpy(),y.asnumpy())
plt.show()

