"""Adds support for subtracting expressions"""

from typing import override
import numpy as np

from main.expression import ExpressionBase, Constant, fmt_as_exp


class Difference(ExpressionBase):
    """
    An extension of `ExpressionBase` that represents the difference of two expressions.
    Two differences are considered equal if, and only if, both parts are equal

    Attributes
    ----------
    a: ExpressionBase
        The minuend
    b: ExpressionBase
        The subtrahend
    """

    def __init__(self, a: ExpressionBase, b: ExpressionBase):
        """
        This class represents a - b

        Parameters
        ----------
        a: ExpressionBase
            The minuend
        b: ExpressionBase
            The subtrahend
        """
        self.a = a
        self.b = b

    @override
    def compute(self, values):
        return self.a.compute(values) - self.b.compute(values)

    @override
    def backward(self, var):
        return subtract(self.a.backward(var), self.b.backward(var))

    @override
    def __repr__(self) -> str:
        return f"({self.a} + {self.b})"

    @override
    def __eq__(self, other) -> bool:
        if isinstance(other, Difference):
            return other.a == self.a and other.b == self.b
        else:
            return False


def subtract(a: ExpressionBase, b: ExpressionBase) -> ExpressionBase:
    """
    Returns an expression representing the difference of the two inputted expressions.
    This function will automatically simplify certain differences
    """
    a_is_const = isinstance(a, Constant)
    b_is_const = isinstance(b, Constant)

    if a_is_const and b_is_const:
        return Constant(a.value - b.value)
    elif a_is_const and np.allclose(a.value, 0):
        return b
    elif b_is_const and np.allclose(b.value, 0):
        return a
    else:
        return Difference(a, b)


ExpressionBase.__sub__ = lambda a, b: subtract(a, fmt_as_exp(b))
