from __future__ import annotations
from typing import Optional, Union, Callable
from math import log, sin, cos


__all__ = ['Variable']

Num = Union[int, float]
Var = Union['Variable', Num]
GradFunc = Optional[Callable[['Variable'], None]]


class VariableDescriptor:
    def __set_name__(self, owner, name):
        self.name = f'_{name}'

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        self._check_value(value)
        setattr(instance, self.name, value)

    @classmethod
    def _check_value(cls, value):
        if not isinstance(value, (int, float)):
            raise ValueError('Numeric data is expected (int, float).')


class Variable:
    value = VariableDescriptor()

    def __init__(
        self,
        value: Num,
        operation: GradFunc = None,
        left_node: Optional[Variable] = None,
        right_node: Optional[Variable] = None,
        requires_grad: bool = True
    ):
        self.value = value
        self.requires_grad = requires_grad
        self.grad = 0 if self.requires_grad else None

        self._operation = operation
        self._left_node = left_node
        self._right_node = right_node

    def __str__(self) -> str:
        return f'Value({self.value}, requires_grad={self.requires_grad})'

    def __add__(self, other: Var) -> Variable:
        return self._binary_general_calc(other, lambda a, b: a + b, Variable._add_backward)

    def __radd__(self, other: Num) -> Variable:
        return self + other

    def __sub__(self, other: Var) -> Variable:
        return self._binary_general_calc(other, lambda a, b: a - b, Variable._sub_backward)

    def __rsub__(self, other: Num) -> Variable:
        return -self + other

    def __mul__(self, other: Var) -> Variable:
        return self._binary_general_calc(other, lambda a, b: a * b, Variable._mul_backward)

    def __rmul__(self, other: Num) -> Variable:
        return self * other

    def __pow__(self, other: Var) -> Variable:
        return self._binary_general_calc(other, lambda a, b: a ** b, Variable._pow_backward)

    def __rpow__(self, other: Num) -> Variable:
        return Variable(other, requires_grad=False) ** self

    def __truediv__(self, other: Var) -> Variable:
        return self._binary_general_calc(other, lambda a, b: a / b, Variable._truediv_backward)

    def __rtruediv__(self, other: Num) -> Variable:
        return self ** -1 * other

    def __neg__(self) -> Variable:
        return self * -1

    def sin(self) -> Variable:
        return self._unary_general_calc(sin, Variable._sin_backward)

    def cos(self) -> Variable:
        return self._unary_general_calc(cos, Variable._cos_backward)

    def _binary_general_calc(
            self,
            second_node: Var,
            operation: Callable[[Num, Num], Num],
            backward_operation: GradFunc
    ) -> Variable:
        if not isinstance(second_node, Variable):
            second_node = Variable(second_node)

        requires_grad = self.requires_grad | second_node.requires_grad
        return Variable(
            operation(self.value, second_node.value),
            operation=backward_operation if requires_grad else None,
            left_node=self,
            right_node=second_node,
            requires_grad=requires_grad
        )

    def _unary_general_calc(
        self,
        operation: Callable[[Num], Num],
        backward_operation: GradFunc
    ) -> Variable:
        return Variable(
            operation(self.value),
            operation=backward_operation if self.requires_grad else None,
            left_node=self,
            right_node=None,
            requires_grad=self.requires_grad
        )

    def _general_grad_calc(
            self,
            left_derivative: Num,
            right_derivative: Num = 0
    ) -> None:
        if self._left_node.requires_grad:
            self._left_node.grad += left_derivative * self.grad
        if self._right_node and self._right_node.requires_grad:
            self._right_node.grad += right_derivative * self.grad

    def _add_backward(self) -> None:
        self._general_grad_calc(1, 1)

    def _sub_backward(self) -> None:
        self._general_grad_calc(1, -1)

    def _mul_backward(self) -> None:
        self._general_grad_calc(
            self._right_node.value,
            self._left_node.value
        )

    def _pow_backward(self) -> None:
        left_val = self._left_node.value
        right_val = self._right_node.value

        self._general_grad_calc(
            right_val * left_val ** (right_val - 1),
            left_val ** right_val * log(left_val) if left_val > 0 else float('nan')
        )

    def _truediv_backward(self) -> None:
        self._general_grad_calc(1 / self._right_node.value,
                                -self._left_node.value / self._right_node.value ** 2)

    def _sin_backward(self) -> None:
        self._general_grad_calc(cos(self._left_node.value))

    def _cos_backward(self) -> None:
        self._general_grad_calc(-sin(self._left_node.value))

    def backward(self) -> None:
        self.grad = 1

        stack = [self]
        while stack:
            curr_node = stack.pop()

            if curr_node and curr_node.requires_grad and curr_node._operation:
                stack.append(curr_node._left_node)
                stack.append(curr_node._right_node)

                curr_node._operation(curr_node)

    def zero_grad(self):
        if not self.requires_grad:
            return

        self.grad = 0
