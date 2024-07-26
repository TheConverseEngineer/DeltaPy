"""Adds support for dividing expressions"""

from typing import override
import numpy as np

from main.expression import ExpressionBase, Constant, fmt_as_exp
from main.operations.subtraction import subtract
from main.operations.multiplication import multiply
from main.operations.power import power, Power


class Quotient(ExpressionBase):
    """
    An extension of `ExpressionBase` that represents the product of two expressions.
    Two products are considered equal if, and only if, both parts are equal


    Attributes
    ----------
    a: ExpressionBase
      The divident
    b: ExpressionBase
      The divisor
    """

    def __init__(self, a: ExpressionBase, b: ExpressionBase):
        self.a = a
        self.b = b

    @override
    def compute(self, values):
        return np.divide(self.a.compute(values), self.b.compute(values))

    @override
    def backward(self, var):
        return divide(
            subtract(
                multiply(self.a, self.b.backward(var)),
                multiply(self.b, self.a.backward(var)),
            ),
            power(self.b, 2),
        )

    @override
    def __repr__(self) -> str:
        return f"({self.a}/{self.b})"

    @override
    def __eq__(self, other) -> bool:
        if isinstance(other, Quotient):
            return other.a == self.a and other.b == self.b
        else:
            return False


def divide(a: ExpressionBase, b: ExpressionBase) -> ExpressionBase:
    """
    Returns an expression representing the quotient of the two inputted expressions.
    This function will automatically simplify certain quotient
    """
    a_is_const = isinstance(a, Constant)
    b_is_const = isinstance(b, Constant)

    if a_is_const and b_is_const:
        return Constant(np.divide(a.value, b.value))
    elif a_is_const and np.allclose(a.value, 0):
        return Constant(0)
    elif b_is_const and np.allclose(b.value, 1):
        return a
    elif a == b:
        return Constant(1)
    elif isinstance(a, Power) and a.has_same_base(b):
        return Power(a.base, a.pow - (b.pow if isinstance(b, Power) else 1))
    elif isinstance(b, Power) and b.has_same_base(a):
        return Power(b.base, (a.pow if isinstance(a, Power) else 1) - b.pow)
    else:
        return Quotient(a, b)


ExpressionBase.__div__ = lambda a, b: divide(a, fmt_as_exp(b))
