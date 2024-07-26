"""Adds support for multiplying two expressions together"""

from typing import override

import numpy as np

from main.expression import ExpressionBase, Constant, fmt_as_exp
from main.operations.addition import add

# Importing exponentiation will supply these values
POWER_FUNC = None
POWER_CLASS = None


def set_power_func(func):
    """Internal function that defines a reference to a power function"""
    global POWER_FUNC
    POWER_FUNC = func


def set_power_class(cls):
    """Internal function that defines a reference to a power class"""
    global POWER_CLASS
    POWER_CLASS = cls


class Product(ExpressionBase):
    """
    An extension of `ExpressionBase` that represents the product of two expressions.
    Two products are considered equal if, and only if, both factors are equal

    Attributes
    ----------
    a: ExpressionBase
        The first factor
    b: ExpressionBase
        The second factor
    """

    def __init__(self, a: ExpressionBase, b: ExpressionBase):
        """
        Parameters
        ----------
        a: ExpressionBase
            The first factor
        b: ExpressionBase
            The second factor
        """
        self.a = a
        self.b = b

    @override
    def compute(self, values):
        return self.a.compute(values) * self.b.compute(values)

    @override
    def backward(self, var):
        return add(
            multiply(self.a.backward(var), self.b),
            multiply(self.b.backward(var), self.a),
        )

    @override
    def __repr__(self) -> str:
        return f"({self.a} * {self.b})"

    @override
    def __eq__(self, other) -> bool:
        if isinstance(other, Product):
            return (other.a == self.a and other.b == self.b) or (
                other.a == self.b and other.b == self.a
            )
        else:
            return False


def multiply(a: ExpressionBase, b: ExpressionBase) -> ExpressionBase:
    """
    Returns an expression representing the product of the two inputted expressions.
    This function will automatically simplify certain products
    """
    a_is_const = isinstance(a, Constant)
    b_is_const = isinstance(b, Constant)

    if a_is_const and b_is_const:
        return Constant(a.value * b.value)
    elif (a_is_const and np.allclose(a.value, 0)) or (
        b_is_const and np.allclose(b.value, 0)
    ):
        return Constant(0)
    elif a_is_const and np.allclose(a.value, 1):
        return b
    elif b_is_const and np.allclose(b.value, 1):
        return a

    if POWER_FUNC is not None:
        a_is_power = isinstance(a, POWER_CLASS)
        b_is_power = isinstance(b, POWER_CLASS)

        if (a_is_power and a.has_same_base(b)) or (b_is_power and b.has_same_base(a)):
            return POWER_FUNC(
                a.base if a_is_power else b.base,
                (a.pow if a_is_power else 1) + (b.pow if b_is_power else 1),
            )
        elif (not a_is_power) and (not b_is_power) and a == b:
            return POWER_FUNC(a, 2)

    return Product(a, b)


ExpressionBase.__mul__ = lambda a, b: multiply(a, fmt_as_exp(b))


# Specify wildcard import to not include the power function/power class setters, as those are
# meant to only be used internally
__all__ = ["Product", "multiply"]
