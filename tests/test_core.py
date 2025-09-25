import pytest

from math import cos, sin, log, isnan
from autodiff import Variable


def test_zero_grad():
    x = Variable(10)
    f = x + 10
    f.backward()
    assert x.grad == 1

    x.zero_grad()
    assert x.grad == 0

    x = Variable(10, requires_grad=None)
    x.zero_grad()
    assert x.grad is None


def test_add_radd():
    x = Variable(2)
    y = Variable(3)
    f1 = x + 2
    f2 = 2 + y
    f1.backward()
    f2.backward()
    assert x.grad == y.grad and x.grad == 1
    assert f1.value == 4 and f2.value == 5

    x.zero_grad()
    y.zero_grad()
    f3 = x + y
    f3.backward()
    assert x.grad == y.grad and x.grad == 1
    assert f3.value == 5


def test_sub_rsub():
    x = Variable(2)
    y = Variable(3)
    f1 = x - 2
    f2 = 2 - y
    f1.backward()
    f2.backward()
    assert x.grad == 1 and y.grad == -1
    assert f1.value == 0 and f2.value == -1

    x.zero_grad()
    y.zero_grad()
    f3 = x - y
    f3.backward()
    assert x.grad == 1 and y.grad == -1
    assert f3.value == -1


def test_mul_rmul():
    x = Variable(2)
    y = Variable(3)
    f1 = x * 2
    f2 = 2 * y
    f1.backward()
    f2.backward()
    assert x.grad == y.grad and x.grad == 2
    assert f1.value == 4 and f2.value == 6

    x.zero_grad()
    y.zero_grad()
    f3 = x * y
    f3.backward()
    assert x.grad == y.value and y.grad == x.value
    assert f3.value == 6


def test_pow_rpow():
    x = Variable(2)
    y = Variable(3)
    f1 = x ** 2
    f2 = 2 ** y
    f1.backward()
    f2.backward()
    assert x.grad == 2 * x.value and y.grad == 2 ** y.value * log(2)
    assert f1.value == 4 and f2.value == 8

    x.zero_grad()
    y.zero_grad()
    f3 = x ** y
    f3.backward()
    assert x.grad == y.value * \
        x.value ** (y.value - 1) and y.grad == x.value ** y.value * \
        log(x.value)
    assert f3.value == 8


def test_nan_pow():
    x = Variable(2)
    f = (-2) ** x
    f.backward()
    assert isnan(x.grad)


def test_truediv_rtruediv():
    x = Variable(8)
    y = Variable(4)
    f1 = x / 2
    f2 = 2 / y
    f1.backward()
    f2.backward()
    assert x.grad == 1 / 2 and y.grad == -2 / y.value ** 2
    assert f1.value == 4 and f2.value == 1 / 2

    x.zero_grad()
    y.zero_grad()
    f3 = x / y
    f3.backward()
    assert x.grad == 1 / y.value and y.grad == -x.value / y.value ** 2
    assert f3.value == 2


def test_neg():
    x = Variable(5)
    f = -x
    f.backward()
    assert x.grad == -1
    assert f.value == -5


def test_sin_cos():
    x = Variable(10)
    y = Variable(5)
    f1 = x.sin()
    f2 = y.cos()
    f1.backward()
    f2.backward()
    assert x.grad == cos(x.value) and y.grad == -sin(y.value)
    assert f1.value == sin(x.value) and f2.value == cos(y.value)


def test_grad_false():
    x = Variable(5, requires_grad=False)
    f = x + 2
    f.backward()
    assert x.grad is None


def test_division_by_zero():
    x = Variable(5)
    with pytest.raises(ZeroDivisionError):
        result = x / 0 # noqa


def test_non_num():
    with pytest.raises(ValueError):
        x = Variable('a') # noqa


def test_example1():
    x = Variable(2)
    y = Variable(3)
    f = x * y + x ** 2 - 1 / y
    f.backward()
    assert x.grad == 7 and y.grad == (2 + 1 / 9)


def test_two_example2():
    x = Variable(10)
    y = Variable(3, requires_grad=False)
    f = x * 10 + y
    f.backward()
    assert x.grad == 10 and y.grad is None


def test_example3():
    x = Variable(3)
    y = Variable(2)
    f = x ** y + y ** x
    f.backward()
    assert x.grad == (y.value * x.value ** (y.value - 1)) + \
        y.value ** x.value * log(y.value)


def test_example4():
    x = Variable(2)
    y = Variable(3)
    f = x.cos() * y.sin()
    f.backward()
    assert x.grad == -sin(x.value) * sin(y.value) and y.grad == cos(x.value) * cos(y.value)
    assert f.value == cos(x.value) * sin(y.value)
