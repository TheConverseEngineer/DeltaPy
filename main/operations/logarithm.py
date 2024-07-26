"""Adds support for taking the logarithm of certain expressions"""

from typing import override
import numpy as np

from main.expression import ExpressionBase, Constant
from main.operations.multiplication import multiply
from main.operations.division import divide


class Logarithm(ExpressionBase):
    """
    An extension of `ExpressionBase` that represents a logarithm with a constant base

    Attributes
    ----------
    base: float | np.ndarray
        The base of this logarithm
    argument: ExpressionBase
        The argument of this logarithm
    """

    def __init__(self, base: float | np.ndarray, argument: ExpressionBase):
        """
        Parameters
        ----------
        base: float | np.ndarray
            The base of this logarithm. Note that using expressions
            as bases is not currently supported.
        argument: ExpressionBase
            The argument of this logarithm.
        """
        self.base = base
        self.argument = argument

    @override
    def compute(self, values):
        return np.log(self.argument.compute(values)) / np.log(self.base)

    @override
    def backward(self, var):
        return divide(
            multiply(self.argument.backward(var), Constant(np.log(self.base))),
            self.argument,
        )

    @override
    def __repr__(self) -> str:
        if np.allclose(self.base, np.e):
            return f"ln{self.argument}"
        else:
            return f"log_({self.base}){self.argument}"

    @override
    def __eq__(self, other) -> bool:
        if isinstance(other, Logarithm):
            return np.allclose(self.base, other.base) and self.argument == other.argument
        return False


def ln(arg: ExpressionBase):
    """Returns the natural logarithm of an expression"""
    if isinstance(arg, Constant):
        return Constant(np.log(arg.value))
    else:
        return Logarithm(np.e, arg)


def log(base: float | np.ndarray, arg: ExpressionBase):
    """Returns the logarithm of an expression with the given base"""
    if isinstance(arg, Constant):
        return Constant(np.log(arg.value) / np.log(base))
    else:
        return Logarithm(base, arg)
