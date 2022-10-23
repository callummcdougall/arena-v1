import torch as t
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

import re
import warnings
import numpy as np
import pandas as pd

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol, Union
# from matplotlib import pyplot as plt

MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")
Arr = np.ndarray
grad_tracking_enabled = True

@dataclass(frozen=True)
class Recipe:
    """Extra information necessary to run backpropagation. You don't need to modify this."""

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll create for it later on."
    args: tuple
    "The input arguments passed to func."
    kwargs: dict[str, Any]
    "Keyword arguments passed to func. To keep things simple today, we aren't going to backpropagate with respect to these."
    parents: dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position, in order to be able to pass gradients back along the computational graph."

class Tensor:
    """
    A drop-in replacement for torch.Tensor supporting a subset of features.
    """

    array: Arr
    "The underlying array. Can be shared between multiple Tensors."
    requires_grad: bool
    "If True, calling functions or methods on this tensor will track relevant data for backprop."
    grad: Optional["Tensor"]
    "Backpropagation will accumulate gradients into this field."
    recipe: Optional[Recipe]
    "Extra information necessary to run backpropagation."

    def __init__(self, array: Union[Arr, list], requires_grad=False):
        self.array = array if isinstance(array, Arr) else np.array(array)
        self.requires_grad = requires_grad
        self.grad = None
        self.recipe = None
        "If not None, this tensor's array was created via recipe.func(*recipe.args, **recipe.kwargs)."

    def __neg__(self) -> "Tensor":
        return negative(self)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __eq__(self, other):
        return eq(self, other)

    def __repr__(self) -> str:
        return f"Tensor({repr(self.array)}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if self.array.ndim == 0:
            raise TypeError
        return self.array.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, index) -> "Tensor":
        return getitem(self, index)

    def add_(self, other: "Tensor", alpha: float = 1.0) -> "Tensor":
        add_(self, other, alpha=alpha)
        return self

    @property
    def T(self) -> "Tensor":
        return permute(self)

    def item(self):
        return self.array.item()

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def reshape(self, new_shape):
        return reshape(self, new_shape)

    def expand(self, new_shape):
        return expand(self, new_shape)

    def permute(self, dims):
        return permute(self, dims)

    def maximum(self, other):
        return maximum(self, other)

    def relu(self):
        return relu(self)

    def argmax(self, dim=None, keepdim=False):
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low: float, high: float) -> "Tensor":
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad: Union[Arr, "Tensor", None] = None) -> None:
        if isinstance(end_grad, Arr):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        """Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html"""
        if self.requires_grad and self.recipe and self.recipe.parents:
            return False
        return True

    def __bool__(self):
        if np.array(self.shape).prod() != 1:
            raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
        return bool(self.item())


def empty(*shape: int) -> Tensor:
    """Like torch.empty."""
    return Tensor(np.empty(shape))


def zeros(*shape: int) -> Tensor:
    """Like torch.zeros."""
    return Tensor(np.zeros(shape))


def arange(start: int, end: int, step=1) -> Tensor:
    """Like torch.arange(start, end)."""
    return Tensor(np.arange(start, end, step=step))


def tensor(array: Arr, requires_grad=False) -> Tensor:
    """Like torch.tensor."""
    return Tensor(array, requires_grad=requires_grad)

def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backwards function for f(x) = log(x)

    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x). Provided as an optimization in case it's cheaper to express the gradient in terms of the output.
    x: the input of np.log.

    Return: gradient of the given loss wrt x
    """
    return grad_out / x

def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    """Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    """
    
    # sum and remove prepended dims
    n_dims_to_sum = len(broadcasted.shape) - len(original.shape)
    broadcasted = broadcasted.sum(axis=tuple(range(n_dims_to_sum)))
    
    # sum over dims that were originally 1
    dims_to_sum = tuple([
        i for i, (o, b) in enumerate(zip(original.shape, broadcasted.shape))
        if o == 1 and b > 1
    ])
    broadcasted = broadcasted.sum(axis=dims_to_sum, keepdims=True)
    
    return broadcasted



def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Union[Arr, float]) -> Arr:
    """Backwards function for x * y wrt argument 0 aka x."""
    if not isinstance(y, Arr):
        y = np.array(y)
    return unbroadcast(y * grad_out, x)


def multiply_back1(grad_out: Arr, out: Arr, x: Union[Arr, float], y: Arr) -> Arr:
    """Backwards function for x * y wrt argument 1 aka y."""
    if not isinstance(x, Arr):
        x = np.array(x)
    return unbroadcast(x * grad_out, y)


def forward_and_back(a: Arr, b: Arr, c: Arr) -> tuple[Arr, Arr, Arr]:
    """
    Calculates the output of the computational graph above (g), then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    """
    d = a * b
    e = np.log(c)
    f = d * e
    g = np.log(f)
    
    final_grad_out = np.array([1.0])
    dg_df = log_back(grad_out=final_grad_out, out=g, x=f)
    dg_dd = multiply_back0(dg_df, f, d, e)
    dg_de = multiply_back1(dg_df, f, d, e)
    dg_da = multiply_back0(dg_dd, d, a, b)
    dg_db = multiply_back1(dg_dd, d, a, b)
    dg_dc = log_back(dg_de, e, c)
    
    return (dg_da, dg_db, dg_dc)


class BackwardFuncLookup:
    def __init__(self) -> None:
        self.d = defaultdict(dict)

    def add_back_func(self, forward_fn: Callable, arg_position: int, back_fn: Callable) -> None:
        self.d[forward_fn][arg_position] = back_fn

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        return self.d[forward_fn][arg_position]


BACK_FUNCS = BackwardFuncLookup()
BACK_FUNCS.add_back_func(np.log, 0, log_back)
BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)


def log_forward(x: Tensor) -> Tensor:
    
    array = np.log(x.array)
    requires_grad = grad_tracking_enabled and (x.requires_grad or (x.recipe is not None))
    out = Tensor(array, requires_grad)
    
    if requires_grad:
        out.recipe = Recipe(func=np.log, args=(x.array,), kwargs={}, parents={0: x})
    else:
        out.recipe = None
        
    return out

log = log_forward


def multiply_forward(a: Union[Tensor, int], b: Union[Tensor, int]) -> Tensor:
    assert isinstance(a, Tensor) or isinstance(b, Tensor)
    
    # Deal with cases where a, b are ints or Tensors, then calculate output
    arg_a = a.array if isinstance(a, Tensor) else a
    arg_b = b.array if isinstance(b, Tensor) else b
    out_arr = arg_a * arg_b
    
    # Find whether the tensor requires grad (need to check if ANY of the inputs do)
    requires_grad = grad_tracking_enabled and any([
        (isinstance(x, Tensor) and (x.requires_grad or x.recipe is not None)) for x in (a, b)
    ])
    
    # Define our tensor
    out = Tensor(out_arr, requires_grad)
    
    # If requires_grad, then create a recipe
    if requires_grad:
        parents = {idx: arr for idx, arr in enumerate([a, b]) if isinstance(arr, Tensor)}
        out.recipe = Recipe(np.multiply, (arg_a, arg_b), {}, parents)
        
    return out

multiply = multiply_forward


def wrap_forward_fn(numpy_func: Callable, is_differentiable=True) -> Callable:
    """
    numpy_func: function. It takes any number of positional arguments, some of which may be NumPy arrays, and any number of keyword arguments which we aren't allowing to be NumPy arrays at present. It returns a single NumPy array.
    is_differentiable: if True, numpy_func is differentiable with respect to some input argument, so we may need to track information in a Recipe. If False, we definitely don't need to track information.

    Return: function. It has the same signature as numpy_func, except wherever there was a NumPy array, this has a Tensor instead.
    """

    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        
        arg_arrays = [(a.array if isinstance(a, Tensor) else a) for a in args]
        out_arr = numpy_func(*arg_arrays, **kwargs)
        
        requires_grad = grad_tracking_enabled and is_differentiable and any([
            (isinstance(a, Tensor) and (a.requires_grad or a.recipe is not None)) for a in args
        ])
        
        out = Tensor(out_arr, requires_grad)
        
        if requires_grad:
            parents = {idx: a for idx, a in enumerate(args) if isinstance(a, Tensor)}
            out.recipe = Recipe(numpy_func, arg_arrays, kwargs, parents)
            
        return out

    return tensor_func


class Node:
    def __init__(self, *children):
        self.children = list(children)

def get_children(node: Node) -> list[Node]:
    return node.children

        
def topological_sort(node: Node, get_children_fn: Callable) -> list[Any]:
    """
    Return a list of node's descendants in reverse topological order from future to past.
    
    Should raise an error if the graph with `node` as root is not in fact acyclic.
    """
    
    perm: set[Node] = set() # stores the nodes which have already been added to `result`
    temp: set[Node] = set() # keeps track of previously visited nodes (to detect cyclicity)
    result: list[Node] = [] # stores the list of nodes to be returned (in reverse topological order)

    # 
    def visit(cur: Node):
        """
        Recursive function which visits all the children of the current node
        """
        if cur in perm:
            return
        if cur in temp:
            raise ValueError("Not a DAG!")
        temp.add(cur)

        for next in get_children_fn(cur):
            visit(next)

        temp.remove(cur)
        perm.add(cur)
        result.append(cur)

    visit(node)
    return result



def sorted_computational_graph(node: Tensor) -> list[Tensor]:
    """
    For a given tensor, return a list of Tensors that make up the nodes of the given Tensor's computational graph, in reverse topological order.
    """
    
    def get_parents(node: Tensor) -> list[Tensor]:
        if node.recipe is None:
            return []
        return list(node.recipe.parents.values())

    return topological_sort(node, get_parents)[::-1]


def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
    """Accumulates gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node: 
        The rightmost node in the computation graph. 
        If it contains more than one element, end_grad must be provided.
    end_grad: 
        A tensor of the same shape as end_node. 
        Set to 1 if not specified and end_node has only one element.
    """
    
    # Get value of end_grad_arr
    end_grad_arr = np.ones_like(end_node.array) if end_grad is None else end_grad.array
    
    # Create dict to store gradients
    grads: dict[Tensor, Arr] = {end_node: end_grad_arr}

    # Iterate through the computational graph, using your sorting function
    for node in sorted_computational_graph(end_node):
        
        # Get the outgradient (recall we need it in our backward functions)
        outgrad = grads.pop(node)
        # We only store the gradients if this node is a leaf (see the is_leaf property of Tensor)
        if node.is_leaf:
            # Add the gradient to this node's grad (need to deal with special case grad=None)
            if node.grad is None:
                node.grad = Tensor(outgrad)
            else:
                node.grad.array += outgrad
                
        # If node has no recipe, then it has no parents, i.e. the backtracking through computational
        # graph ends here
        if node.recipe is None:
            continue
            
        # If node has a recipe, then we iterate through parents (which is a dict of {arg_posn: tensor})
        for argnum, parent in node.recipe.parents.items():
            
            # Get the backward function corresponding to the function that created this node,
            # and the arg posn of this particular parent within that function 
            back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)
            
            # Use this backward function to calculate the gradient
            in_grad = back_fn(outgrad, node.array, *node.recipe.args, **node.recipe.kwargs)
            
            # Add the gradient to this node in the dictionary `grads`
            # Note that we only change the grad of the node itself in the code block above
            if grads.get(parent) is None:
                grads[parent] = in_grad
            else:
                grads[parent] += in_grad


def _argmax(x: Arr, dim=None, keepdim=False):
    """Like torch.argmax."""
    # Changing from "keepdims" because this is throwing errors.
    return np.expand_dims(np.argmax(x, axis=dim), axis=([] if dim is None else dim))

argmax = wrap_forward_fn(_argmax, is_differentiable=False)
eq = wrap_forward_fn(np.equal, is_differentiable=False)


def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backward function for f(x) = -x elementwise."""
    return np.full_like(x, -1) * grad_out

negative = wrap_forward_fn(np.negative)
BACK_FUNCS.add_back_func(np.negative, 0, negative_back)


def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return out * grad_out

exp = wrap_forward_fn(np.exp)
BACK_FUNCS.add_back_func(np.exp, 0, exp_back)


def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return np.reshape(grad_out, x.shape)

reshape = wrap_forward_fn(np.reshape)
BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)


def invert_transposition(axes: tuple) -> tuple:
    """
    axes: tuple indicating a transition
    
    Returns: inverse of this transposition, i.e. the array `axes_inv` s.t. we have:
        np.transpose(np.transpose(x, axes), axes_inv) == x
    
    Some examples:
        (1, 0)    --> (1, 0)     # this is reversing a simple 2-element transposition
        (0, 2, 1) --> (0, 1, 2)
        (1, 2, 0) --> (2, 0, 1)  # this is reversing the order of a 3-cycle
    """
    
    # Slick solution:
    return tuple(np.argsort(axes))

    # Slower solution, which makes it clearer what operation is happening:
    reversed_transposition_map = {num: idx for (idx, num) in enumerate(axes)}
    reversed_transposition = [reversed_transposition_map[idx] for idx in range(len(axes))]
    return tuple(reversed_transposition)

def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    return np.transpose(grad_out, invert_transposition(axes))

BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
permute = wrap_forward_fn(np.transpose)


def expand_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return unbroadcast(grad_out, x)

def _expand(x: Arr, new_shape) -> Arr:
    """Like torch.expand, calling np.broadcast_to internally.

    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    """
    n_added = len(new_shape) - x.ndim
    shape_non_negative = tuple([x.shape[i - n_added] if s == -1 else s for i, s in enumerate(new_shape)])
    return np.broadcast_to(x, shape_non_negative)

expand = wrap_forward_fn(_expand)
BACK_FUNCS.add_back_func(_expand, 0, expand_back)


def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False):
    """Basic idea: repeat grad_out over the dims along which x was summed"""
    
    # If grad_out is a scalar, we need to make it a tensor (so we can expand it later)
    if not isinstance(grad_out, Arr):
        grad_out = Tensor(grad_out)
    
    # If dim=None, this means we summed over all axes, and we want to repeat back to input shape
    if dim is None:
        dim = list(range(x.ndim))
        
    # If keepdim=False, then we need to add back in dims, so grad_out and x have same number of dims
    if keepdim == False:
        grad_out = np.expand_dims(grad_out, dim)
    
    # Finally, we repeat grad_out along the dims over which x was summed
    return np.broadcast_to(grad_out, x.shape)


def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    """Like torch.sum, calling np.sum internally."""
    return np.sum(x, axis=dim, keepdims=keepdim)

sum = wrap_forward_fn(_sum)

BACK_FUNCS.add_back_func(_sum, 0, sum_back)


Index = Union[int, tuple[int, ...], tuple[Arr], tuple[Tensor]]

def coerce_index(index: Index) -> Union[int, tuple[int, ...], tuple[Arr]]:
    """
    If index is of type signature `tuple[Tensor]`, converts it to `tuple[Arr]`.
    """
    if isinstance(index, tuple) and set(map(type, index)) == {Tensor}:
        return tuple([i.array for i in index])
    else:
        return index

def _getitem(x: Arr, index: Index) -> Arr:
    """Like x[index] when x is a torch.Tensor."""
    return x[coerce_index(index)]

def getitem_back(grad_out: Arr, out: Arr, x: Arr, index: Index):
    """Backwards function for _getitem.

    Hint: use np.add.at(a, indices, b)
    This function works just like a[indices] += b, except that it allows for repeated indices.
    """
    new_grad_out = np.full_like(x, 0)
    np.add.at(new_grad_out, coerce_index(index), grad_out)
    return new_grad_out

getitem = wrap_forward_fn(_getitem)
BACK_FUNCS.add_back_func(_getitem, 0, getitem_back)


add = wrap_forward_fn(np.add)
subtract = wrap_forward_fn(np.subtract)
true_divide = wrap_forward_fn(np.true_divide)

BACK_FUNCS.add_back_func(np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))
BACK_FUNCS.add_back_func(np.add, 1, lambda grad_out, out, x, y: unbroadcast(grad_out, y))
BACK_FUNCS.add_back_func(np.subtract, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))
BACK_FUNCS.add_back_func(np.subtract, 1, lambda grad_out, out, x, y: unbroadcast(-grad_out, y))
BACK_FUNCS.add_back_func(np.true_divide, 0, lambda grad_out, out, x, y: unbroadcast(grad_out/y, x))
BACK_FUNCS.add_back_func(np.true_divide, 1, lambda grad_out, out, x, y: unbroadcast(grad_out*(-x/y**2), y))


def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
    """Like torch.add_. Compute x += other * alpha in-place and return tensor."""
    np.add(x.array, other.array * alpha, out=x.array)
    return x


def safe_example():
    """This example should work properly."""
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    a.add_(b)
    c = a * b
    c.sum().backward()
    assert a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0])
    assert b.grad is not None and np.allclose(b.grad.array, [2.0, 4.0, 6.0, 8.0])


def unsafe_example():
    """This example is expected to compute the wrong gradients."""
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    c = a * b
    a.add_(b)
    c.sum().backward()
    if a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0]):
        print("Grad wrt a is OK!")
    else:
        print("Grad wrt a is WRONG!")
    if b.grad is not None and np.allclose(b.grad.array, [0.0, 1.0, 2.0, 3.0]):
        print("Grad wrt b is OK!")
    else:
        print("Grad wrt b is WRONG!")


def maximum_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    """Backwards function for max(x, y) wrt x."""
    bool_sum = ((x > y) + 0.5 * (x == y))
    return unbroadcast(grad_out * bool_sum, x)
    # MLAB solution:
    # out = grad_out.copy()
    # out = np.where(x == y, grad_out / 2, out)
    # out = np.where(x < y, 0.0, out)
    # return unbroadcast(out, x)

def maximum_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    """Backwards function for max(x, y) wrt y."""
    bool_sum = ((x < y) + 0.5 * (x == y))
    return unbroadcast(grad_out * bool_sum, y)

maximum = wrap_forward_fn(np.maximum)
BACK_FUNCS.add_back_func(np.maximum, 0, maximum_back0)
BACK_FUNCS.add_back_func(np.maximum, 1, maximum_back1)



def relu(x: Tensor) -> Tensor:
    """Like torch.nn.function.relu(x, inplace=False)."""
    return maximum(x, 0.0)


def _matmul2d(x: Arr, y: Arr) -> Arr:
    """Matrix multiply restricted to the case where both inputs are exactly 2D."""
    return x @ y

def matmul2d_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    return grad_out @ y.T

def matmul2d_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    return x.T @ grad_out

matmul = wrap_forward_fn(_matmul2d)
BACK_FUNCS.add_back_func(_matmul2d, 0, matmul2d_back0)
BACK_FUNCS.add_back_func(_matmul2d, 1, matmul2d_back1)


class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True):
        """Share the array with the provided tensor."""
        self.array = tensor.array
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"


class Module:
    _modules: dict[str, "Module"]
    _parameters: dict[str, Parameter]

    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def modules(self):
        """Return the direct child modules of this module."""
        return self.__dict__["_modules"].values()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over Module parameters.

        recurse: if True, the iterator includes parameters of submodules, recursively.
        """
        parameters_list = list(self.__dict__["_parameters"].values())
        if recurse:
            for mod in self.modules():
                parameters_list.extend(list(mod.parameters(recurse=True)))
        return iter(parameters_list)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        If val is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call the superclass.
        """
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Union[Parameter, "Module"]:
        """
        If key is in _parameters or _modules, return the corresponding value.
        Otherwise, raise KeyError.
        """
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]
        
        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        
        raise KeyError(key)

        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError("Subclasses must implement forward!")

    def __repr__(self):
        def _indent(s_, numSpaces):
            return re.sub("\n", "\n" + (" " * numSpaces), s_)
        lines = [f"({key}): {_indent(repr(module), 2)}" for key, module in self._modules.items()]
        return "".join([
            self.__class__.__name__ + "(",
            "\n  " + "\n  ".join(lines) + "\n" if lines else "", ")"
        ])


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # sf needs to be a float
        sf = in_features ** -0.5
        
        weight = sf * Tensor(2 * np.random.rand(out_features, in_features) - 1)
        self.weight = Parameter(weight)
        
        if bias:
            bias = sf * Tensor(2 * np.random.rand(out_features,) - 1)
            self.bias = Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        out = x @ self.weight.permute((1, 0))
        if self.bias is not None: 
            out += self.bias
        return out

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"




def cross_entropy(logits: Tensor, true_labels: Tensor) -> Tensor:
    """Like torch.nn.functional.cross_entropy with reduction='none'.

    logits: shape (batch, classes)
    true_labels: shape (batch,). Each element is the index of the correct label in the logits.

    Return: shape (batch, ) containing the per-example loss.
    """
    n_batch, n_class = logits.shape
    true = logits[arange(0, n_batch), true_labels]
    return -log(exp(true) / exp(logits).sum(1))