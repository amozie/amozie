from functools import reduce, wraps


def foo1(*args):
    for arg in args:
        print(arg)


def foo2(**kwargs):
    for key, value in kwargs.items():
        print('{key} == {value}'.format(**locals()))


def fibon(n):
    i, a, b = 1, 1, 2
    while i <= n:
        yield a
        a, b = b, a + b
        i += 1


def a_new_decorator(a_func):
    @wraps(a_func)
    def wrapTheFuncion():
        print('start')
        a_func()
        print('end')

    return wrapTheFuncion


def a_func():
    print('this is a func')


@a_new_decorator
def a_func_dec():
    a_func()


def func_dec(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        print('start')
        f(*args, **kwargs)
        print('end')

    return decorated


@func_dec
@func_dec
def func_be_dec(n):
    print(n)


if __name__ == '__main__':
    l = range(4)
    d = dict(a=1, b=2, c=3, d=4)
    foo1(*l)
    foo2(**d)
    for i, v in enumerate(fibon(5)):
        print(i + 1, v)
    print([i * i for i in range(10)])
    print([i for i in range(-5, 5) if i < 0])
    print(reduce((lambda x, y: x * y), [1, 2, 3, 4]))
    print(1 if True else 2)
    dec = a_new_decorator(a_func)
    dec()
    a_func_dec()
    print(a_func_dec.__name__)
    func_be_dec(1)
