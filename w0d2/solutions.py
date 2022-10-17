# Section 1

from fancy_einsum import einsum
from typing import Union, Optional, Callable
import numpy as np

def einsum_trace(mat):
    return einsum("i i", mat)

def einsum_mv(mat, vec):
    return einsum("i j, j -> i", mat, vec)

def einsum_mm(mat1, mat2):
    return einsum("i j, j k -> i k", mat1, mat2)

def einsum_inner(vec1, vec2):
    return einsum("i, i", vec1, vec2)

def einsum_outer(vec1, vec2):
    return einsum("i, j -> i j", vec1, vec2)



# Section 2

import torch as t
from collections import namedtuple
TestCase = namedtuple("TestCase", ["output", "size", "stride"])

test_cases = [
    TestCase(
        output=t.tensor([0, 1, 2, 3]), 
        size=(4,), 
        stride=(1,)
    ),
    # Explanation: the output is a 1D vector of length 4 (hence size=(4,))
    # and each time you move one element along in this output vector, you also want to move
    # one element along the `test_input_a` tensor

    TestCase(
        output=t.tensor([0, 1, 2, 3, 4]), 
        size=(5,), 
        stride=(1,)
    ),
    # Explanation: the tensor is held in a contiguous memory block. When you get to the end
    # of one row, a single stride jumps to the start of the next row

    TestCase(
        output=t.tensor([0, 5, 10, 15]), 
        size=(4,), 
        stride=(5,)
    ),
    # Explanation: this is same as previous case, only now you're moving in colspace (i.e. skipping
    # 5 elements) each time you move one element across the output tensor.
    # So stride is 5 rather than 1

    TestCase(
        output=t.tensor([[0, 1, 2], [5, 6, 7]]), 
        size=(2, 3), 
        stride=(5, 1)),
    # Explanation: consider the output tensor. As you move one element along a row, you want to jump
    # one element in the `test_input_a` (since you're just going to the next row). As you move
    # one element along a column, you want to jump to the next column, i.e. a stride of 5.

    TestCase(
        output=t.tensor(
            [[0, 1, 2], 
             [10, 11, 12]]
        ), 
        size=(2, 3), 
        stride=(10, 1)),

    TestCase(
        output=t.tensor(
            [[0, 0, 0], 
             [11, 11, 11]]
        ), 
        size=(2, 3),
        stride=(11, 0)),

    TestCase(
        output=t.tensor(
            [0, 6, 12, 18]
        ), 
        size=(4,), 
        stride=(6,)),

    TestCase(
        output=t.tensor(
            [[[0, 1, 2]], [[9, 10, 11]]]
        ), 
        size=(2, 1, 3), 
        stride=(9, 0, 1)),
    # Note here that the middle element of `stride` doesn't actually matter, since you never
    # jump in this dimension. You could change it and the test result would still be the same

    TestCase(
        output=t.tensor(
            [
                [
                    [[0, 1], [2, 3]], 
                    [[4, 5], [6, 7]]
                ], 
                [
                    [[12, 13], [14, 15]], 
                    [[16, 17], [18, 19]]
                ]
            ]
        ),
        size=(2, 2, 2, 2),
        stride=(12, 4, 2, 1),
    ),
]




def as_strided_trace(mat: t.Tensor) -> t.Tensor:
    
    stride = mat.stride()
    
    assert len(stride) == 2, f"matrix should have size 2"
    assert mat.size(0) == mat.size(1), "matrix should be square"
    
    return mat.as_strided((mat.size(0),), (sum(stride),)).sum()

def as_strided_mv(mat: t.Tensor, vec: t.Tensor) -> t.Tensor:
    
    sizeM = mat.shape
    sizeV = vec.shape
    
    strideM = mat.stride()
    strideV = vec.stride()
    
    assert len(sizeM) == 2, f"mat1 should have size 2"
    assert sizeM[1] == sizeV[0], f"mat{list(sizeM)}, vec{list(sizeV)} not compatible for multiplication"
    
    vec_expanded = vec.as_strided(mat.shape, (0, strideV[0]))
    
    product_expanded = mat * vec_expanded
    
    return product_expanded.sum(dim=1)

def as_strided_mm(matA: t.Tensor, matB: t.Tensor) -> t.Tensor:
    
    assert len(matA.shape) == 2, f"mat1 should have size 2"
    assert len(matB.shape) == 2, f"mat2 should have size 2"
    assert matA.shape[1] == matB.shape[0], f"mat1{list(matA.shape)}, mat2{list(matB.shape)} not compatible for multiplication"
    
    # Get the matrix strides, and matrix dims
    sA0, sA1 = matA.stride()
    dA0, dA1 = matA.shape
    sB0, sB1 = matB.stride()
    dB0, dB1 = matB.shape
    
    expanded_size = (dA0, dA1, dB1)
    
    matA_expanded_stride = (sA0, sA1, 0)
    matA_expanded = matA.as_strided(expanded_size, matA_expanded_stride)
    
    matB_expanded_stride = (0, sB0, sB1)
    matB_expanded = matB.as_strided(expanded_size, matB_expanded_stride)
    
    product_expanded = matA_expanded * matB_expanded
    
    return product_expanded.sum(dim=1)





def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    
    batch, in_channels, width = x.shape
    out_channels, in_channels_2, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"
    output_width = width - kernel_width + 1
    
    xsB, xsI, xsWi = x.stride()
    wsO, wsI, wsW = weights.stride()
    
    x_new_shape = (batch, in_channels, output_width, kernel_width)
    x_new_stride = (xsB, xsI, xsWi, xsWi)
    # Common error: xsWi is always 1, so if you put 1 here you won't spot your mistake until you try this with conv2d!
    x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)
    
    return einsum(
        "batch in_channels output_width kernel_width, out_channels in_channels kernel_width -> batch out_channels output_width", 
        x_strided, weights
    )

def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """
    
    batch, in_channels, height, width = x.shape
    out_channels, in_channels_2, kernel_height, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"
    output_width = width - kernel_width + 1
    output_height = height - kernel_height + 1
    
    xsB, xsIC, xsH, xsW = x.stride() # B for batch, IC for input channels, H for height, W for width
    wsOC, wsIC, wsH, wsW = weights.stride()
    
    x_new_shape = (batch, in_channels, output_height, output_width, kernel_height, kernel_width)
    x_new_stride = (xsB, xsIC, xsH, xsW, xsH, xsW)
    
    x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)
    
    return einsum(
        "batch in_channels output_height output_width kernel_height kernel_width, \
out_channels in_channels kernel_height kernel_width \
-> batch out_channels output_height output_width",
        x_strided, weights
    )

def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """
    B, C, W = x.shape
    output = x.new_full(size=(B, C, left + W + right), fill_value=pad_value)
    output[..., left : left + W] = x
    # Note - you can't use `left:-right`, because `right` could be zero.
    return output
    


def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    B, C, H, W = x.shape
    output = x.new_full(size=(B, C, top + H + bottom, left + W + right), fill_value=pad_value)
    output[..., top : top + H, left : left + W] = x
    return output

def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    """Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    
    x_padded = pad1d(x, left=padding, right=padding, pad_value=0)
    
    batch, in_channels, width = x_padded.shape
    out_channels, in_channels_2, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"
    output_width = 1 + (width - kernel_width) // stride
    # note, we assume padding is zero in the formula here, because we're working with input which has already been padded
    
    xsB, xsI, xsWi = x_padded.stride()
    wsO, wsI, wsW = weights.stride()
    
    x_new_shape = (batch, in_channels, output_width, kernel_width)
    x_new_stride = (xsB, xsI, xsWi * stride, xsWi)
    # Explanation for line above:
    #     we need to multiply the stride corresponding to the `output_width` dimension
    #     because this is the dimension that we're sliding the kernel along
    x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)
    
    return einsum("B IC OW wW, OC IC wW -> B OC OW", x_strided, weights)

IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    """Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    """

    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    
    x_padded = pad2d(x, left=padding_w, right=padding_w, top=padding_h, bottom=padding_h, pad_value=0)
    
    batch, in_channels, height, width = x_padded.shape
    out_channels, in_channels_2, kernel_height, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"
    output_width = 1 + (width - kernel_width) // stride_w
    output_height = 1 + (height - kernel_height) // stride_h
    
    xsB, xsIC, xsH, xsW = x_padded.stride() # B for batch, IC for input channels, H for height, W for width
    wsOC, wsIC, wsH, wsW = weights.stride()
    
    x_new_shape = (batch, in_channels, output_height, output_width, kernel_height, kernel_width)
    x_new_stride = (xsB, xsIC, xsH * stride_h, xsW * stride_w, xsH, xsW)
    
    x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)
    
    return einsum("B IC OH OW wH wW, OC IC wH wW -> B OC OH OW", x_strided, weights)


def maxpool2d(x: t.Tensor, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0
) -> t.Tensor:
    """Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    """

    if stride is None:
        stride = kernel_size
    stride_height, stride_width = force_pair(stride)
    padding_height, padding_width = force_pair(padding)
    kernel_height, kernel_width = force_pair(kernel_size)
    
    x_padded = pad2d(x, left=padding_width, right=padding_width, top=padding_height, bottom=padding_height, pad_value=-t.inf)
    
    batch, channels, height, width = x_padded.shape
    output_width = 1 + (width - kernel_width) // stride_width
    output_height = 1 + (height - kernel_height) // stride_height
    
    xsB, xsC, xsH, xsW = x_padded.stride()
    
    x_new_shape = (batch, channels, output_height, output_width, kernel_height, kernel_width)
    x_new_stride = (xsB, xsC, xsH * stride_height, xsW * stride_width, xsH, xsW)
    
    x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)
    
    output = t.amax(x_strided, dim=(-1, -2))
    return output


# =============== PART 4 ===============

from torch import nn

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Call the functional version of maxpool2d."""
        return maxpool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])


class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0.0))


import functools
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape
        
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        
        shape_left = shape[:start_dim]
        shape_middle = functools.reduce(lambda x, y: x*y, shape[start_dim : end_dim+1])
        shape_right = shape[end_dim+1:]
        
        new_shape = shape_left + (shape_middle,) + shape_right
        
        return t.reshape(input, new_shape)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        sf = 1 / np.sqrt(in_features)
        
        weight = sf * (2 * t.rand(out_features, in_features) - 1)
        self.weight = nn.Parameter(weight)
        
        if bias:
            bias = sf * (2 * t.rand(out_features,) - 1)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        x = einsum("... in_features, out_features in_features -> ... out_features", x, self.weight)
        if self.bias is not None: x += self.bias
        return x

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        kernel_height, kernel_width = force_pair(kernel_size)
        sf = 1 / np.sqrt(in_channels * kernel_width * kernel_height)
        weight = sf * (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1)
        self.weight = nn.Parameter(weight)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d you wrote earlier."""
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])