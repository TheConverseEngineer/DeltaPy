# DeltaPy

A lightweight autodifferentiation and backpropagation library written in python using numpy. 

## What this does

Autodifferentiation refers to dynamically keeping track of expressions to automatically calculate their partial derivates. Consider the following expression: $f(x)=3xy+z^2$. In python, we could manually calculate the value for this function as follows.

```python
x = 15
y = 12
z = 4

f = 3*x*y + z**2
dfdx = 3*y
dfdz = 2*z
```

As you can see, not only does this approach require us to define each variable in advance, but all derivatives must be manually calculated (which is tedious when using larger functions). Now observe the approach using the DeltaPy library.

```python
from main.expression import Variable
from main.operations import *

x = Variable(x)
y = Variable(y)
z = Variable(z)

f = x*y*3 + z**2
```

DeltaPy takes advantage of operator overloading to keep usage simple and consise. Internally, the variable `f` maintains a graph representing the expression $3xy+z^2$. Thanks to this, we can now compute the value of this any function 
and take derivatives easily.

```python
v1 = f.compute({x: 15, y: 12, z: 4})    # Returns f(15, 12, 4)

# Automatically computes the derivative of f with respect to x
dfdx = f.backward(x)       

# Chaining this function will compute higher order derivatives
dfdz2 = f.backward(z).backward(z)

# Returns the second partial derivative of f with respect to z at (15, 12, 4)
v2 = dfdz2.compute({x: 15, y: 12, z: 4}) 
```

## How to use this library

Simply add the main module to any existing python project. To access the base classes of `Variable` and `Constant`, import them as follows:

```python
from main.expression import Variable, Constant
```

To take advantage of operator overloads, simply import the desired operators from `main.operations` or use the wildcard import to import all operations. In addition to overloaded operations, the library also features the following functions:

```python
from main.operations import *

# Natural log
logarithm.ln(expression)

# Logarithm with a custom base
logarithm.log(base, expression)

# Exponent functions
exponent.exp(expression) # Uses e as the base
exponent.exponent(base, expression)
```

In order to compute the value of any expression, use the `compute` method with a dictionary of keys mapping each variable to a value. Values can be either in the form of floats or numpy arrays. To calculate the partial derivative of a function, use the `backward` method and input the variable that the partial derivate should be taken with respect to.
