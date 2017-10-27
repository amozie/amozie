import numpy as np


def foo(n):
    print(list(range(n)))


if __name__ == '__main__':
    x = 1
    y = 2
    z = x + y
    x = z
    print(z)
