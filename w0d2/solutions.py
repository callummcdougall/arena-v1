# Section 1

from fancy_einsum import einsum

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