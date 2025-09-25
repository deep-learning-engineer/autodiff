# AutoDiff - Automatic Differentiation Library

AutoDiff is a Python library for automatic differentiation, designed to simplify gradient computations for machine learning and mathematical optimization. It provides a simple and intuitive interface for defining mathematical expressions and automatically computing their derivatives.

## Features

- 🚀 **Forward-mode automatic differentiation**
- 📐 **Support for basic arithmetic operations** (+, -, *, /, **)
- 🔢 **Trigonometric function support** (sin, cos, ...)
- 🧮 **Multi-variable differentiation**
- 💻 **Pure Python implementation** - no external dependencies
- 🧪 **Comprehensive test suite**


## Usage Examples

### Basic Arithmetic

```python
from autodiff import Variable

# Single variable differentiation
x = Variable(5.0)
f = x**3 + 3*x**2 - 72*x + 90
f.backward()

print(f"f(5) = {f.value}")  # Output: -70.0
print(f"f'(5) = {x.grad}")  # Output: 33.0
```

### Trigonometric Functions

```python
from autodiff import Variable
from math import pi

# Trigonometric operations
x = Variable(pi/4)
y = Variable(pi/3)

f = x.sin() + y.cos()
f.backward()

print(f"sin(π/4) + cos(π/3) = {f.value}") # Output: 1.207...
print(f"∂f/∂x = {x.grad}")  # Output: 0.707...
print(f"∂f/∂y = {y.grad}")  # Output: -0.866...
```

### Multi-variable Calculus

```python
from autodiff import Variable

# Multi-variable function
x = Variable(2.0)
y = Variable(3.0)
z = Variable(4.0)

f = (x * y) + (y * z) + (z * x)
f.backward()

print(f"Function value: {f.value}")  # Output: 26.0
print(f"Gradient: ({x.grad}, {y.grad}, {z.grad})")  # Output: (7.0, 6.0, 5.0)
```

## API Reference

### Variable Class

The `Variable` class represents a value with optional gradient tracking.

```python
Variable(value, requires_grad=True)
```

- `value`: Initial numeric value (int or float)
- `is_grad`: Boolean flag to enable gradient tracking

### Supported Operations

- Arithmetic: `+`, `-`, `*`, `/`, `**`, `neg`
- Trigonometric: `sin()`, `cos()`
- Method: `backward()` - computes gradients for all variables

## Implementation Details

AutoDiff uses a computational graph approach to automatically track operations and compute derivatives using the chain rule. The library implements both forward and reverse mode automatic differentiation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you have any questions or issues, please open an issue on the GitHub repository.