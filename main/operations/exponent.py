"""Adds support for raising constants to an expression as a power"""

from typing import override
import numpy as np

from main.expression import ExpressionBase, Constant
from main.operations.multiplication import multiply


class Exponent(ExpressionBase):
    """
    An extension of `ExpressionBase` that represents a constant raised to an expression as
    a power

    Attributes
    ----------
    base: float | np.ndarray
        The base of this exponent
    pow: ExpressionBase
        The argument of this exponent
    """

    def __init__(self, base: float | np.ndarray, power: ExpressionBase):
        """
        Parameters
        ----------
        base: float | np.ndarray
            The base of this exponent. Note that using expressions
            as bases is not currently supported.
        pow: ExpressionBase
            The power of this exponent.
        """
        self.base = base
        self.power = power

    @override
    def compute(self, values):
        return np.power(self.base, self.power.compute(values))

    @override
    def backward(self, var):
        return multiply(
            Constant(np.log(self.base)),
            multiply(exponent(self.base, self.power), self.power.backward(var)),
        )

    @override
    def __repr__(self) -> str:
        if np.allclose(self.base, np.e):
            return f"e^{self.power}"
        else:
            return f"{self.base}^{self.power}"

    @override
    def __eq__(self, other) -> bool:
        if isinstance(other, Exponent):
            return np.allclose(self.base, other.base) and self.power == other.power
        return False


def exponent(base: float | np.ndarray, power: ExpressionBase):
    """
    Raises a constant to an expression as a power

    Parameters
    ----------
    base: float | np.ndarray
        The base of this exponent.
    pow: ExpressionBase
        The power of this exponent.
    """
    if isinstance(power, Constant):
        return Constant(np.power(base, power.value))
    elif np.allclose(base, 0):
        return Constant(0)
    elif np.allclose(base, 1):
        return Constant(1)
    else:
        return Exponent(base, power)


def exp(power: ExpressionBase):
    """
    Raises e to an expression as a power

    Parameters
    ----------
    pow: ExpressionBase
        The power of this exponent.
    """
    if isinstance(power, Constant):
        return Constant(np.exp(power.value))
    else:
        return Exponent(np.e, power)
