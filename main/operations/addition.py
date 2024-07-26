"""Adds support for adding expressions"""

from typing import override
import numpy as np

from main.expression import ExpressionBase, Constant, fmt_as_exp


class Sum(ExpressionBase):
    """
    An extension of `ExpressionBase` that represents the sum of two expressions.
    Two sums are considered equal if, and only if, both addends are equal


    Attributes
    ----------
    a: ExpressionBase
        The first addend
    b: ExpressionBase
        The second addend
    """

    def __init__(self, a: ExpressionBase, b: ExpressionBase):
        """
        Parameters
        ----------
        a: ExpressionBase
            The first addend
        b: ExpressionBase
            The second addend
        """
        self.a = a
        self.b = b

    @override
    def compute(self, values):
        return self.a.compute(values) + self.b.compute(values)

    @override
    def backward(self, var):
        return add(self.a.backward(var), self.b.backward(var))

    @override
    def __repr__(self) -> str:
        return f"({self.a} + {self.b})"

    @override
    def __eq__(self, other) -> bool:
        if isinstance(other, Sum):
            return (other.a == self.a and other.b == self.b) or (
                other.a == self.b and other.b == self.a
            )
        else:
            return False


def add(a: ExpressionBase, b: ExpressionBase) -> ExpressionBase:
    """
    Returns an expression representing the sum of the two inputted expressions.
    This function will automatically simplify certain sums
    """

    a_is_const = isinstance(a, Constant)
    b_is_const = isinstance(b, Constant)

    if a_is_const and b_is_const:
        return Constant(a.value + b.value)
    elif a_is_const and np.allclose(a.value, 0):
        return b
    elif b_is_const and np.allclose(b.value, 0):
        return a
    else:
        return Sum(a, b)


ExpressionBase.__add__ = lambda a, b: add(a, fmt_as_exp(b))
