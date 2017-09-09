import numpy as np
import matplotlib.pyplot as plt


class Regression():
    def __init__(self, x, y, add_constant=True, theta=None):
        self.x, self.y, self.theta = _pretreatment_params(x, y, add_constant, theta)

    def _hypothesis(self, x, theta):
        pass

    def _cost_function(self, theta):
        return _square_cost_function(self._hypothesis, self.y, self.x, theta)

    def gradient_descent(self, learning_rate=0.1, max_iteration=2000, delta_theta_rel=1e-6):
        print('starting gradient descent...')
        last_cost = self._cost_function(self.theta)
        current_iteration = 0
        while current_iteration < max_iteration:
            current_iteration += 1
            current_theta = []
            for i, theta_i in enumerate(self.theta):
                theta_i -= learning_rate * _partial_derivative(self._cost_function, self.theta, i, learning_rate)
                current_theta.append(theta_i)
            last_theta = self.theta
            current_theta = np.array(current_theta)
            self.theta = current_theta

            current_cost = self._cost_function(self.theta)
            yield current_cost

            delta_cost_actual = last_cost - current_cost
            if delta_cost_actual < 0:
                print("warning: on iteration {0} cost function increased".format(current_iteration))
            last_cost = current_cost

            if np.max(np.abs((current_theta - last_theta) / last_theta)) < delta_theta_rel:
                break
        print("complete on iteration {0}".format(current_iteration))
        print("theta: {0}".format(self.theta))

    def predict(self, x):
        x = np.array(x)
        if x.size + 1 == self.theta.size:
            x = np.insert(x, 0, [1])
        return self._hypothesis(x, self.theta)


class LinearRegression(Regression):
    def _hypothesis(self, x, theta):
        return linear_hypothesis(x, theta)

    def normal_equation(self):
        self.theta = np.dot(np.dot(np.linalg.pinv(np.dot(self.x.T, self.x)), self.x.T), self.y)
        return self.theta


class LogisticRegression(Regression):
    def _hypothesis(self, x, theta):
        return logistic_hypothesis(x, theta)


def linear_hypothesis(x, theta):
    return np.dot(x, theta)


def logistic_hypothesis(x, theta):
    return 1 / (1 + np.exp(-np.dot(x, theta)))


def _pretreatment_params(x, y, add_constant=True, theta=None):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(x.size, 1)
    if add_constant:
        x = _add_constant(x)
    if y and y.ndim != 1:
        y = y.reshape(y.size)
    if not theta:
        theta = np.random.randn(x.shape[1])
    return x, y, theta


def _add_constant(x):
    return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)


def _partial_derivative(cost_function, theta, i, learing_rate):
    theta_pos_delta = theta.copy()
    theta_pos_delta[i] += learing_rate
    theta_neg_delta = theta.copy()
    theta_neg_delta[i] -= learing_rate
    return (cost_function(theta_pos_delta) - cost_function(theta_neg_delta)) / 2 / learing_rate


def _square_cost_function(hypothesis, y, x, theta):
    return np.sum((hypothesis(x, theta) - y) ** 2) / 2 / y.size


if __name__ == '__main__':
    x = np.array([1.1, 2.1, 2.9, 3.9, 5.2])
    y = np.array([3, 5, 7, 9, 11])
    reg = LinearRegression(x, y)
    fig = plt.subplot(111)

    for it in reg.gradient_descent():
        print(it)

    print(reg.normal_equation())
    print(reg.predict(1))
