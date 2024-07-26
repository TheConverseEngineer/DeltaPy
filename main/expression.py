"""
Includes the abstract class `ExpressionBase`, used as the base for all expressions, along with the
two fundamental expression types: `Variable` and `Constant`
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, override
import numpy as np


class ExpressionBase(ABC):
    """
    Abstract class that serves as the base for all operations/expressions
    All operations extend this class and implement their respective functionality
    This allows Expressions to be recursively nested inside one another
    """

    @abstractmethod
    def compute(self, values: Dict[Variable, float | np.ndarray]) -> float | np.ndarray:
        """
        Returns this expression, evaluated at the given values

        Parameters
        ----------
        values: Dict[Variable, float | np.ndarray]
            A dictionary of variables and their values at the point where this
            expression should be evaluated. If a value is represented as a numpy arrays,
            all operations will be conducted element-wise

        Returns
        -------
        float | np.ndarray
            The evaluated point, with the same type as the type provided for
            each value in `values`
        """

    @abstractmethod
    def backward(self, var: Variable) -> ExpressionBase:
        """
        Returns an expression that represents the derivative of this expression

        Parameters
        ----------
        var: Variable
            The variable to differentiate with respect to

        Returns
        -------
        ExpressionBase
            An expression representing the derivative of this expression with
            respect to `var`
        """


class Variable(ExpressionBase):
    """
    An extension of `ExpressionBase` that represents a variable.

    Note that the variable name is only used for debugging/printing purposes.
    Two variables with the same name are NOT considered equal unless they point
    to the same instance.

    Attributes
    ----------
    name: str
    The name of this variable
    """

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name: str
            The name of this variable (as it should be printed)
        """
        self.name = name

    @override
    def compute(self, values):
        return values[self]

    @override
    def backward(self, var):
        return Constant(1) if var == self else Constant(0)

    @override
    def __repr__(self) -> str:
        return self.name


class Constant(ExpressionBase):
    """
    An extension of `ExpressionBase` that represents a constant value.

    Unlike variables, two constants with the same value (within a tolerance) will
    be considered equal.

    Attributes
    ----------
    value: float | np.ndarray
    The value of this constant
    """

    def __init__(self, value: float | np.ndarray):
        """
        Parameters
        ----------
        value: float | np.ndarray
            The value of this constant,
        """
        self.value = value

    @override
    def compute(self, values):
        return self.value

    @override
    def backward(self, var):
        return Constant(0)

    @override
    def __repr__(self) -> str:
        return f"{self.value}"

    @override
    def __eq__(self, other):
        if isinstance(other, Constant):
            return np.allclose(self.value, other.value)
        return False


def fmt_as_exp(a: float | np.ndarray | ExpressionBase) -> ExpressionBase:
    """
    Turns a float or numpy array into a `Constant` with the same value.
    Inputting an expression will simply return the expression
    """
    if isinstance(a, ExpressionBase):
        return a
    return Constant(a)
