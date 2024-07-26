"""Adds support for raising expressions to constant powers"""

from typing import override
import numpy as np

from main.expression import ExpressionBase, Variable, Constant
from main.operations.multiplication import multiply, set_power_func, set_power_class


class Power(ExpressionBase):
    """
    An extension of `ExpressionBase` that represents an expression raised to a constant power

    Attributes
    ----------
    base: ExpressionBase
        The base of this exponent
    pow: float | np.ndarray
        The power that `base` should be raised to
    """

    def __init__(self, base: ExpressionBase, _power: float | np.ndarray):
        """
        Parameters
        ----------
        base: ExpressionBase
            The base of this exponent
        pow: float | np.ndarray
            The power that `base` should be raised to. Note that expressions
            as powers are not currently supported.
        """
        self.base = base
        self.power = _power

    @override
    def compute(self, values):
        return np.power(self.base.compute(values), self.power)

    @override
    def backward(self, var: Variable):
        return multiply(
            multiply(
                Constant(self.power),
                self.base.backward(var),
            ),
            power(self.base, self.power - 1),
        )

    def has_same_base(self, other) -> bool:
        """
        Returns true if `other` has the same base as this exponent (including the case where
        `other` is not of type `Exponent` and should be treated as being to the first power)
        """
        if isinstance(other, Power):
            return self.base == other.base
        else:
            return self.base == other

    @override
    def __repr__(self) -> str:
        return f"({self.base}^({self.power}))"

    @override
    def __eq__(self, other) -> bool:
        if isinstance(other, Power):
            return other.base == self.base and other.power == self.power


def power(base: ExpressionBase, _power: float | np.ndarray) -> ExpressionBase:
    """
    Returns an expression representing the inputted expression raised to the given power.
    This function will automatically simplify certain powers
    """
    if np.allclose(_power, 1):
        return base
    elif np.allclose(_power, 0):
        return Constant(1)

    if isinstance(base, Constant):
        return Constant(np.power(base.value, _power))
    elif isinstance(base, Power):
        return Power(base.base, base.power * _power)

    return Power(base, _power)


ExpressionBase.__pow__ = power

# Supply multiplication with power functions
set_power_func(power)
set_power_class(Power)

print("done")
