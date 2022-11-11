import streamlit as st

import plotly.io as pio
import re
import json

import platform
is_local = (platform.processor() != "")
rootdir = "" if is_local else "ch0/"

def read_from_html(filename):
    filename = rootdir + f"images/{filename}.html"
    with open(filename) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}    
    return pio.from_json(json.dumps(plotly_json))

def get_fig_dict():
    return {str(i): read_from_html(f"fig{i}") for i in range(1, 16)}

if "fig_dict" not in st.session_state:
    fig_dict = get_fig_dict()
    st.session_state["fig_dict"] = fig_dict
else:
    fig_dict = st.session_state["fig_dict"]

st.set_page_config(layout="wide")

# code > span.string {
#     color: red !important;
# }

st.markdown("""
<style>
label.effi0qh3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 15px;
}
p {
    line-height:1.48em;
}
.streamlit-expanderHeader {
    font-size: 1em;
    color: darkblue;
}
.css-ffhzg2 .streamlit-expanderHeader {
    color: lightblue;
}
header {
    background: rgba(255, 255, 255, 0) !important;
}
code {
    color: red;
    white-space: pre-wrap !important;
}
code:not(h1 code):not(h2 code):not(h3 code):not(h4 code) {
    font-size: 13px;
}
a.contents-el > code {
    color: black;
    background-color: rgb(248, 249, 251);
}
.css-ffhzg2 a.contents-el > code {
    color: orange;
    background-color: rgb(26, 28, 36);
}
.css-ffhzg2 code:not(pre code) {
    color: orange;
}
.css-ffhzg2 .contents-el {
    color: white !important;
}
pre code {
    font-size:13px !important;
}
.katex {
    font-size:17px;
}
h2 .katex, h3 .katex, h4 .katex {
    font-size: unset;
}
ul.contents {
    line-height:1.3em; 
    list-style:none;
    color-black;
    margin-left: -10px;
}
ul.contents a, ul.contents a:link, ul.contents a:visited, ul.contents a:active {
    color: black;
    text-decoration: none;
}
ul.contents a:hover {
    color: black;
    text-decoration: underline;
}
</style>""", unsafe_allow_html=True)

def section0():
    st.markdown("""## 1️⃣ Einops and Einsum

In part 1, we'll go through some basic `einops` and `einsum` exercises. These are designed to get you comfortable with using Einstein summation convention, and why it's useful in tensors. 

This section should take approximately **1 hour** (maybe less if you're already familiar with summation convention)

## 2️⃣ Array strides

`as_strided` is a low-level array method that forces you to engage with the messy details of linear operations like matrix multiplictions and convolutions. Although you probably won't need to use them on a regular basis, they will form an essential part of your toolbox as you build up a neural network from scratch.

This section should take approximately **1-2 hours**.

## 3️⃣ Convolutions

Convolutions are a vital part of image classifiers. In part 3, has you write your own functions to perform 1D and 2D convolutions, using your knowledge of `einsum` and `as_strided` from previous sections.

This section should take **1-2 hours**. If you're confident you understand the basic mechanism of convolutions, and you're pushed for time, you can skip some of the questions in this section.

## 4️⃣ Making your own modules

In part 4, we start on some of the exercises that will be built on in day 3. We'll be taking our functions from sections 2 & 3, and using them to create modules which inherit from PyTorch's `nn.Module`. 

Don't worry if you don't get through all of this part; today's exercises are quite long and it's more important to understand them deeply than to rush ahead! If you do get to this section, it should take you approximately **2 hours**.
""")

def section1():

    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#reading">Reading</a></li>
    <li><a class="contents-el" href="#einops">Einops</a></li>
    <li><a class="contents-el" href="#einsum">Einsum</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""
## Reading

* Read about the benefits of the `einops` library [here](https://www.blopig.com/blog/2022/05/einops-powerful-library-for-tensor-operations-in-deep-learning/).
* If you haven't already, then review the [Einops basics tutorial](https://einops.rocks/1-einops-basics/) (up to the "fancy examples" section).
* Read [einsum is all you need](https://rockt.github.io/2018/04/30/einsum) for a brief overview of the `einsum` function and how it works.

## Einops

First, run this cell to import the libraries and define the objects you'll need:

```python
import numpy as np
from fancy_einsum import einsum
from einops import reduce, rearrange, repeat
from typing import Union, Optional, Callable
import torch as t
import torchvision
import utils

arr = np.load("numbers.npy")
```

`arr` is a 4D numpy array. The first axes corresponds to the number, and the next three axes are channels (i.e. RGB), height and width respectively. You have the function `utils.display_array_as_img` which takes in a numpy array and displays it as an image. There are two possible ways this function can be run:

* If the input is three-dimensional, the dimensions are interpreted as `(channel, height, width)` - in other words, as an RGB image.
* If the input is two-dimensional, the dimensions are interpreted as `(height, width)` - i.e. a monochrome image.

For example:

```python
display_array_as_img(arr[0])
```

produces the following output:""")

    st.plotly_chart(fig_dict["1"], use_container_width=False, config=dict(displayModeBar=False))

    st.markdown("""A series of images follow below, which have been created using `einops` functions performed on `arr`. You should work through these and try to produce each of the images yourself. This page also includes solutions, but you should only look at them after you've tried for at least five minutes.""")

    st.markdown("### Exercise 1")
    st.plotly_chart(fig_dict["2"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Solution"):
        st.code("""arr2 = rearrange(arr, "b c h w -> c h (b w)")""")

    st.markdown("### Exercise 2")
    st.plotly_chart(fig_dict["3"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Solution"):
        st.code("""arr2 = repeat(arr[0], "c h w -> c (2 h) w")""")

    st.markdown("### Exercise 3")
    st.plotly_chart(fig_dict["4"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Solution"):
        st.code("""arr2 = repeat(arr[0:2], "b c h w -> c (b h) (2 w)")""")

    st.markdown("### Exercise 4")
    st.plotly_chart(fig_dict["5"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Solution"):
        st.code("""arr2 = repeat(arr[0], "c h w -> c (h 2) w")""")

    st.markdown("### Exercise 5")
    st.plotly_chart(fig_dict["6"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Solution"):
        st.code("""arr2 = rearrange(arr[0], "c h w -> h (c w)")""")

    st.markdown("### Exercise 6")
    st.plotly_chart(fig_dict["7"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Solution"):
        st.code("""arr2 = rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)""")

    st.markdown("### Exercise 7")
    st.plotly_chart(fig_dict["8"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Solution"):
        st.code("""arr2 = reduce(arr.astype(float), "b c h w -> h (b w)", "max").astype(int)""")

    st.markdown("### Exercise 8")
    st.plotly_chart(fig_dict["9"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Hint"):
        st.markdown("NumPy complains when you take the mean over an integer array. You may need to convert into `float` then back to `int` at the end. Use the `astype` array method.")
    with st.expander("Solution"):
        st.code("""arr2 = reduce(arr.astype(float), "b c h w -> h (b w)", "mean").astype(int)""")

    st.markdown("### Exercise 9")
    st.plotly_chart(fig_dict["10"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Solution"):
        st.code("""arr2 = reduce(arr.astype(float), "b c h w -> h w", "min").astype(int)""")

    st.markdown("### Exercise 10")
    st.plotly_chart(fig_dict["11"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Hint"):
        st.markdown("Try to split this into 2 parts. The first part should just involve creating a 3D array corresponding to the image of [0, 1] side by side.")
    with st.expander("Solution"):
        st.code("""arr2 = rearrange(arr[:2], "b c h w -> c h (b w)")

arr3 = rearrange(arr2, "c (h2 h) w -> c h (h2 w)", h2=2)""")

    st.markdown("### Exercise 11")
    st.plotly_chart(fig_dict["12"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Solution"):
        st.code("""arr2 = rearrange(arr[1], "c h w -> c w h")""")

    st.markdown("### Exercise 12")
    st.plotly_chart(fig_dict["13"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Solution"):
        st.code("""arr2 = rearrange(arr, "(b1 b2) c h w -> c (b1 w) (b2 h)", b1=2)""")

    st.markdown("""### Exercise 13
In this exercise, we use **max pooling**. This is a topic we'll dive deeper into later today, but essentially it involves splitting an array up into grid squares and taking the maximum over each of them. Note that the image below is half the standard size for this image. 

You should find the `reduce` function useful here.
""")
    st.plotly_chart(fig_dict["14"], use_container_width=False, config=dict(displayModeBar=False))
    with st.expander("Solution"):
        st.code("""arr2 = reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2)""")

    st.markdown('''

## Einsum

Einsum is a very useful function for performing linear operations, which you'll probably be using a lot during this programme. Although there are many different kinds of operations you can perform, they are all derived from three key rules:

1. Repeating letters in different inputs means those values will be multiplied, and those products will be in the output. 
    * For example, `M = einsum("ij,jk->ik", A, B)` corresponds to the matrix equation $M=AB$.
2. Omitting a letter means that the axis will be summed.
    * For examples, if `x` is a 2D array with shape `(n0, n1)`, then `einsum("ij->i", x)` will be a 1D array of length `n0` containing the row sums of `x`.
3. We can return the unsummed axes in any order.
    * For example, `einsum("ijk->kji", x)` does the same thing as `einops.rearrange(x, "i j k -> k j i")`.

A quick note about `fancy_einsum` before we start - it behaves differently than `einsum`, because of spaces.

For instance, `np.einsum` could use a string like `"ij->i"` to mean "sum a single array over the second dimension", but `fancy_einsum` would get confused because it sees `"ij"` and `"i"` each as single words, referring to individual dimensions. So you'd need `"i j -> i"` in this case. To avoid confusion, it's recommended to only use `fancy_einsum` rather than switching between the two (there will be cases when you'll be thankful for `fancy_einsum`'s features!).

In the following exercises, you'll write simple functions using `einsum` which replicate the functionality of standard NumPy functions: trace, matrix multiplication, inner and outer products. We've also included some test functions which you should run.

```python
def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    pass

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    pass

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    pass

def einsum_inner(vec1, vec2):
    """
    Returns the same as `np.inner`.
    """
    pass

def einsum_outer(vec1, vec2):
    """
    Returns the same as `np.outer`.
    """
    pass

utils.test_einsum_trace(einsum_trace)
utils.test_einsum_mv(einsum_mv)
utils.test_einsum_mm(einsum_mm)
utils.test_einsum_inner(einsum_inner)
utils.test_einsum_outer(einsum_outer)
```
''')

    with st.expander("Help - I get 'TypeError: cannot use a string pattern on a bytes-like object'"):
        st.markdown("""This is probably because you have strings and arrays the wrong way round. In `einsum`, the string goes first and the arrays follow. This is because `einsum` accepts a variable number of arrays but only one string. `einops` functions only work on single arrays, so the array is the first argument for those functions.""")

def section2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#reading">Reading</a></li>
    <li><a class="contents-el" href="#basic-stride-exercises">Basic stride exercises</a></li>
    <li><a class="contents-el" href="#intermediate-stride-exercises">Intermediate stride exercises</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""
## Reading

* [Python NumPy, 6.1 - `as_strided()`](https://www.youtube.com/watch?v=VlkzN00P0Bc) explains what array strides are.
* [`as_strided` and `sum` are all you need](https://jott.live/markdown/as_strided) gives an overview of how to use `as_strided` to perform array operations. 
* [Advanced NumPy: Master stride tricks with 25 illustrated exercises](https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20) provides several clear and intuitive examples of `as_strided` being used to construct arrays.

## Basic stride exercises

Array strides, and the `as_strided` method, are important to understand well because lots of linear operations are actually implementing something like `as_strided` under the hood.

Consider the following tensor:

```python
test_input = t.tensor(
    [[0, 1, 2, 3, 4], 
    [5, 6, 7, 8, 9], 
    [10, 11, 12, 13, 14], 
    [15, 16, 17, 18, 19]], dtype=t.float
)
```

This tensor is stored in a contiguous block in computer memory.

We can call the `stride` method to get the strides of this particular array. Running `test_input.stride()`, we get `(5, 1)`. This means that we need to skip over one element in the storage of this tensor to get to the next element in the row, and 5 elements to get the next element in the column (because you have to jump over all 5 elements in the row). Another way of phrasing this: the `n`th element in the stride is the number of elements we need to skip over to move one index position in the `n`th dimension.

In the exercises below, we will work with the `test_input` tensor above. You should fill in the `size` and `stride` arguments so that calling `test_input.as_strided` with these arguments produces the desired output. When you run the cell, the `for` loop at the end will iterate through the test cases and print out whether the test passed or failed.

We've already filled in the first one as an example. The output is a 1D tensor of length 4 (hence we want `size=(4,)`), and the values are the first row of `input_tensor` (hence we want to move one element along the `input_tensor` at each step, i.e. `stride=1`).

By the end of these examples, hopefully you'll have a clear idea of what's going on. If you're still confused by some of these, then `solutions.py` contains some annotations to explain the answers.

```python
import torch as t
from collections import namedtuple

TestCase = namedtuple("TestCase", ["output", "size", "stride"])

test_cases = [
    TestCase(
        output=t.tensor([0, 1, 2, 3]), 
        size=(4,), 
        stride=(1,)),
    TestCase(
        output=t.tensor([0, 1, 2, 3, 4]), 
        size=None, 
        stride=None),
    TestCase(
        output=t.tensor([0, 5, 10, 15]), 
        size=None, 
        stride=None),
    TestCase(
        output=t.tensor([[0, 1, 2], [5, 6, 7]]), 
        size=None, 
        stride=None),
    TestCase(
        output=t.tensor([[0, 1, 2], [10, 11, 12]]), 
        size=None, 
        stride=None),
    TestCase(
        output=t.tensor([[0, 0, 0], [11, 11, 11]]), 
        size=None,
        stride=None),    
    TestCase(
        output=t.tensor([0, 6, 12, 18]), 
        size=None, 
        stride=None),
    TestCase(
        output=t.tensor(<code>0, 1, 2]], [[9, 10, 11</code>), 
        size=None, 
        stride=None),
    TestCase(
        output=t.tensor(<code>[0, 1], [2, 3]], [[4, 5], [6, 7</code>, <code>12, 13], [14, 15]], [[16, 17], [18, 19</code>]),
        size=None,
        stride=None),
]
for (i, case) in enumerate(test_cases):
    if (case.size is None) or (case.stride is None):
        print(f"Test {i} failed: attempt missing.")
    else:
        actual = test_input.as_strided(size=case.size, stride=case.stride)
        if (case.output != actual).any():
            print(f"Test {i} failed:")
            print(f"Expected: {case.output}")
            print(f"Actual: {actual}\n")
        else:
            print(f"Test {i} passed!\n")
```

## Intermediate stride exercises

Now that you're comfortable with the basics, we'll dive a little deeper with `as_strided`. In the last few exercises of this section, you'll start to implement some more challenging stride functions: trace, matrix-vector and matrix-matrix multiplication, just like we did for `einsum` in the previous section.

```python
def as_strided_trace(mat: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    pass

utils.test_trace(as_strided_trace)
```
""")

    with st.expander("Hint"):
        st.markdown("The trace is the sum of all the elements you get from starting at `[0, 0]` and then continually stepping down and right one element. Use strides to create a 1D array which contains these elements.")

    st.markdown("""
```python
def as_strided_mv(mat: t.Tensor, vec: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    pass

utils.test_mv(as_strided_mv)
utils.test_mv2(as_strided_mv)
```
""")

    with st.expander("Hint 1"):
        st.markdown("""You want your output array to be as follows:
    
```output[i] = sum_j { mat[i, j] * vec[j] }```

so first try to create an array with `arr[i, j] = mat[i, j] * vec[j]`, then we can sum over this to get our output.""")

    with st.expander("Hint 2"):
        st.markdown("""Use striding to create an expanded vector with `vec_expanded[i, j] = vec[j]`, then we can compute `arr` as described in hint 1.""")

    with st.expander("Help - I'm passing the first test, but failing the second."):
        st.markdown("""It's possible that the input matrices you recieve could themselves be the output of an `as_strided` operation, so that they're represented in memory in a non-contiguous way. Make sure that your `as_strided `operation is using the strides from the original input arrays, i.e. it's not just assuming the last element in the `stride()` tuple is 1.""")

    st.markdown("""
```python
def as_strided_mm(matA: t.Tensor, matB: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    pass

utils.test_mm(as_strided_mm)
utils.test_mm2(as_strided_mm)
```
""")

    with st.expander("Hint 1"):
        st.markdown("""
If you did the first one, this isn't too dissimilar. We have:

```output[i, k] = sum_j { matA[i, j] * matB[j, k] }```

so in this case, try to create an array with `arr[i, j, k] = matA[i, j] * matB[j, k]`.

We need to create expanded versions of both `matA` and `matB` in order to take this product.
""")

    with st.expander("Hint 2"):
        st.markdown("""We want `matA_expanded[i, j, k] = matA[i, j]`, so our stride for `matA should be `(matA.stride(0), matA.stride(1), 0)`.
        
A similar idea applies for `matB`.""")

def section3():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#reading">Reading</a></li>
    <li><a class="contents-el" href="#exercise-1-implement-conv1d-minimal">Exercise 1 - Implement <code>conv1d_minimal</code></a></li>
    <li><a class="contents-el" href="#exercise-2-implement-conv2d-minimal">Exercise 2 - Implement <code>conv2d_minimal</code></a></li>
    <li><a class="contents-el" href="#exercise-3-implement-pad1d-and-pad2d">Exercise 3 - Implement <code>pad1d</code> and <code>pad2d</code></a></li>
    <li><a class="contents-el" href="#exercise-4-implement-conv1d-and-conv2d">Exercise 4 - Implement <code>conv1d</code> and <code>conv2d</code></a></li>
    <li><a class="contents-el" href="#exercise-5-implement-maxpool">Exercise 5 - Implement <code>maxpool</code></a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""
## Reading

* [A Comprehensive Guide to Convolutional Neural Networks (TowardsDataScience)](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

Here are some questions to make sure you've understood the material. Once you finish the article above, you should try and answer these questions without referring back to the original article.
""")

    with st.expander("Why would convolutional layers be less likely to overfit data than standard linear (fully connected) layers?"):
        st.markdown("""Convolutional layers require significantly fewer weights to be learned. This is because the same kernel is applied all across the image, rather than every pair of `(input, output)` nodes requiring a different weight to be learned.""")

    with st.expander("Suppose you fixed some random permutation of the pixels in an image, and applied this to all images in your dataset, before training a convolutional neural network for classifying images. Do you expect this to be less effective, or equally effective?"):
        st.markdown("""It will be less effective, because CNNs work thanks to **spatial locality** - groups of pixels close together are more meaningful. For instance, CNNs will often learn convolutions at an early layer which recognise gradients or simple shapes. If you permute the pixels (even if you permute in the same way for every image), you destroy locality. """)

    with st.expander("If you have a 28x28 image, and you apply a 3x3 convolution with stride 1, padding 1, what shape will the output be?"):
        st.markdown("""It will be the same shape, i.e. `28x28`. In the post linked above, this is described as **same padding**. Tomorrow, we'll build an MNIST classifier which uses these convolutions.""")

    st.markdown("""
## Exercise 1 - Implement `conv1d_minimal`

Here, we will implement the PyTorch `conv1d` function, which can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html). We will start with a simple implementation where `stride=1` and `padding=0`, with the other arguments set to their default values.

Firstly, some explanation of `conv1d` in PyTorch. The `1` in `1d` here refers to the number of dimensions along which we slide the weights (also called the kernel) when we convolve. Importantly, it does not refer to the number of dimensions of the tensors that are being used in our calculations. Typically the input and kernel are both 3D:

* `input.shape = (batch, in_channels, width)`
* `kernel.shape = (out_channels, in_channels, kernel_width)`

A typical convolution operation is illustrated in the sketch below. Some notes on this sketch:

* The `kernel_width` dimension of the kernel slides along the `width` dimension of the input. The `output_width` of the output is determined by the number of kernels that can be fit inside it; the formula can be seen in the right part of the sketch.
* For each possible position of the kernel inside the model (i.e. each freezeframe position in the sketch), the operation happening is as follows:
    * We take the product of the kernel values with the corresponding input values, and then take the sum
    * This gives us a single value for each output channel
    * These values are then passed into the output tensor
* The sketch assumes a batch size of 1. To generalise to a larger batch number, we can just imagine this operation being repeated identically on every input.
""")

    st.image(rootdir + "images/conv1d_illustration.png")

    st.markdown("""
Below, you should implement `conv1d_minimal`. This is a function which works just like `conv1d`, but takes the default stride and padding values (these will be added back in later). You are allowed to use `as_strided` and `einsum`.

This is intended to be pretty challenging, so we've provided several hints which you should work through in sequence if you get stuck.
""")

    with st.expander("Hint 1"):
        st.markdown("""First, consider a simplified version where `batch` and `out_channels` are both 1. These are both pretty simple to add back in later, because the convolution operation is done identically along the batch dimension, and each slice of the kernel corresponding to one of the `out_channels` is convolved with `x` in exactly the same way.
        
So we have `x.shape = (in_channels, width)`, and `weights.shape = (in_channels, kernel_width)`, and we want to get output of shape `(output_width,)`.""")

    with st.expander("Hint 2"):
        st.markdown("""We want to get a strided version of `x`, which we can then multiply with `weights` (using `einops`) to get something of the required shape.
        
The shape of `x_strided` should be `(in_channels, output_width, kernel_width)`. Try and think about what each of the strides should be.""")

    with st.expander("Hint 3"):
        st.markdown("""The strides for the first two dimensions of `x_strided` should be the same as `x.stride()`. For the stride corresponding to `kernel_width`, every time we move the kernel one step along inside `x` we also want to move one step inside `x`, so this stride should be `x.stride()[1]`.
        
So we have:

```python
xsB, xsI, xsWi = x.stride()
x_new_stride = (xsB, xsI, xsWi, xsWi)
```

Now try and turn this into a full function. Return to Hint1 if you're confused.""")

    st.markdown("""
```python
def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    pass
    
utils.test_conv1d_minimal(conv1d_minimal)
```

## Exercise 2 - Implement `conv2d_minimal`

2D convolutions are conceptually similar to 1D. The only difference is in how you move the kernel across the tensor as you take your convolution. In this case, you will be moving the tensor across two dimensions:
""")
    st.image(rootdir + "images/conv2d_illustration.png", width=600)

    st.markdown("""
For this reason, 1D convolutions tend to be used for signals (e.g. audio), 2D convolutions are used for images, and 3D convolutions are used for 3D scans (e.g. in medical applications). 

You should implement `conv2d` in a similar way to `conv1d`. Again, this is expected to be difficult and there are several hints you can go through.
""")

    with st.expander("Hint 1"):
        st.markdown("""This is conceptually very similar to conv1d. You can start by copying your code from the conv1d function, but changing it whenever it refers to `width` (since you'll need to use `width` *and* `height`).""")

    with st.expander("Hint 2"):
        st.markdown("""The shape of `x_strided` should be `(batch, in_channels, output_height, output_width, kernel_height, kernel_width)`. 
        
Just like last time, some of these strides should just correspond to their equivalents in `x.stride()`, and you can work out the others by thinking about how the kernel is moved around inside `x`.""")

    st.markdown(r"""
```python
def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    pass
    
utils.test_conv2d_minimal(conv2d_minimal)
```

## Exercise 3 - Implement `pad1d` and `pad2d`

For a full version of `conv`, and for `maxpool` (which will follow shortly), you'll need to implement `pad` helper functions. PyTorch has some very generic padding functions, but to keep things simple and build up gradually, we'll write 1D and 2D functions individually.

Tip: use the `new_full` method of the input tensor. This is a clean way to ensure that the output tensor is on the same device as the input, and has the same dtype.

Tip: you can use three dots to denote slicing over multiple dimensions. For instance, `x[..., 0]` will take the `0th` slice of `x` along its last dimension. This is equivalent to `x[:, 0]` for 2D, `x[:, :, 0]` for 3D, etc.

```python
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    pass


utils.test_pad1d(pad1d)
utils.test_pad1d_multi_channel(pad1d)
```

```python

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    pass

utils.test_pad2d(pad2d)
utils.test_pad2d_multi_channel(pad2d)
```

## Exercise 4 - Implement `conv1d` and `conv2d`

Now extend `conv1d` to handle the `stride` and `padding` arguments.

`stride` is the number of input positions that the kernel slides at each step. `padding` is the number of zeros concatenated to each side of the input before the convolution.

Output shape should be (batch, output_channels, output_length), where output_length can be calculated as follows:

$$
\text{output\_length} = \left\lfloor\frac{\text{input\_length} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} \right\rfloor + 1
$$

Verify for yourself that the forumla above simplifies to the formula we used earlier when padding is 0 and stride is 1.

Docs for pytorch's `conv1d` can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html).""")

    with st.expander("Hint"):
        st.markdown("""Each step of the kernel inside the input tensor, you're moving by `stride` elements rather than just 1 element.
        
So when creating `x_strided`, you should change the `stride` argument at the positions corresponding to the movement of the kernel inside `x`, so that you're jumping over `stride` elements rather than 1.

You will also need a new `output_width` (use the formula in the documentation).""")

    st.markdown("""
```python
def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    '''Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    pass

utils.test_conv1d(conv1d)
```
""")

    st.markdown("""
#### Helper functions for pairs

A recurring pattern in these 2d functions is allowing the user to specify either an int or a pair of ints for an argument: examples are stride and padding. We've provided some type aliases and a helper function to simplify working with these.

```python
IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

# Examples of how this function can be used:
#       force_pair((1, 2))     ->  (1, 2)
#       force_pair(2)          ->  (2, 2)
#       force_pair((1, 2, 3))  ->  ValueError
```

Finally, you can implement a full version of `conv2d`. If you've done the full version of `conv1d`, and you've done `conv2d_minimal`, then this shouldn't be too much trouble.

```python
def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    pass
    
utils.test_conv2d(conv2d)
```

## Exercise 5 - Implement `maxpool`

Before we move to section 4, we'll implement one last function: **max pooling**. You can review the [TowardsDataScience](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) post from earlier to understand max pooling better.

A "max pooling" layer is similar to a convolution in that you have a window sliding over some number of dimensions. The main difference is that there's no kernel: instead of multiplying by the kernel and adding, you just take the maximum.

The way multiple channels work is also different. A convolution has some number of input and output channels, and each output channel is a function of all the input channels. There can be any number of output channels. In a pooling layer, the maximum operation is applied independently for each input channel, meaning the number of output channels is necessarily equal to the number of input channels.

Implement `maxpool2d` using `torch.as_strided` and `torch.amax` (= max over axes) together. Your version should behave the same as the PyTorch version, but only the indicated arguments need to be supported.""")

    with st.expander("Hint"):
        st.markdown("""Conceptually, this is similar to `conv2d`. 
    
In `conv2d`, you had to use `as_strided` to turn the 4D tensor `x` into a 6D tensor `x_strided` (adding dimensions over which you would take the convolution), then multiply this tensor by the kernel and sum over these two new dimensions.

`maxpool2d` is the same, except that you're simply taking max over those dimensions rather than a dot product with the kernel. So you should find yourself able to reuse a lot of code from your `conv2d` function.""")

    st.markdown("""```python
def maxpool2d(x: t.Tensor, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0
) -> t.Tensor:
    '''Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, out_height, output_width)
    '''
    pass

utils.test_maxpool2d(maxpool2d)
```""")

    with st.expander("Help - I'm getting a small number of mismatched elements each time (e.g. between 0 and 5%)."):
        st.markdown("""This is likely because you used an incorrect `pad_value`. In the convolution function, we set `pad_value=0` so these values wouldn't have any effect in the linear transformation. What pad value would make our padded elements "invisible" when we take the maximum?
        
Click on the expander below to reveal the answer.""")

    with st.expander("""Click to reveal the answer to the question posed in the expander above this one."""):
        st.markdown("""<span style="background-color: #31333F">$$-\infty$$</span>""", unsafe_allow_html=True)

def section4():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#subclassing-nn-module">Subclassing nn.Module</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#init-and-forward"><code>__init__</code> and <code>forward</code></a></li>
        <li><a class="contents-el" href="#the-nn-parameter-class">The <code>nn.Parameter</code> class</a></li>
        <li><a class="contents-el" href="#printing-information-with-extra-repr">Printing information with <code>extra_repr</code></a></li>
    </ul></li>
    <li><a class="contents-el" href="#exercise-1-maxpool2d-module">Exercise 1 - <code>MaxPool2d</code> module</a></li>
    <li><a class="contents-el" href="#exercise-2-relu-and-flatten">Exercise 2 - <code>ReLU</code> and <code>Flatten</code></a></li>
    <li><a class="contents-el" href="#exercise-3-linear-module">Exercise 3 - <code>Linear</code> module</a></li>
    <li><a class="contents-el" href="#exercise-4-conv2d-module">Exercise 4 - <code>Conv2d</code> module</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""
## Subclassing `nn.Module`

One of the most basic parts of PyTorch that you will see over and over is the `nn.Module` class (you may have encountered this at the [end of yesterday's exercises](https://arena-w0d1.streamlitapp.com/Basic_Neural_Network)). All types of neural net components inherit from it, from the simplest `nn.Relu` to the most complex `nn.Transformer`. Often, a complex `nn.Module` will have sub-`Module`s which implement smaller pieces of its functionality.

Other common `Module`s  you’ll see include

- `nn.Linear`, for fully-connected layers with or without a bias, like you’d see in an MLP
- `nn.Conv2d`, for a two-dimensional convolution, like you’d see in a CNN
- `nn.Softmax`, which implements the softmax function

The list goes on, including activation functions, normalizations, pooling, attention, and more. You can see all the `Module`s that torch provides [here](https://pytorch.org/docs/stable/nn.html). You can also create your own `Module`s, as we will do often!

The `Module` class provides a lot of functionality, but we’ll only cover a little bit of it here.

In this section, we'll add another layer of abstraction to all the linear operations we've done in previous sections, by packaging them inside `nn.Module` objects.

### `__init__` and `forward`

A subclass of `nn.Module` usually looks something like this:

```python
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, arg1, arg2, ...):
        super().__init__()
        # Initialization code 

    def forward(self, x: t.Tensor) -> t.Tensor:
        # Forward pass code
```

The initialization sets up attributes that will be used for the life of the `Module`, like its parameters, hyperparameters, or other sub-`Module`s it might need to use. These are usually added to the instance with something like `self.attribute = attr`, where `attr` might be provided as an argument. Some modules are simple enough that they don’t need any persistent attributes, and in this case you can skip the `__init__`.

The `forward` method is called on each forward pass of the `Module`, possibly using the attributes that were set up in the `__init__`. It should take in the input, do whatever it’s supposed to do, and return the result. Subclassing `nn.Module` automatically makes instances of your class callable, so you can do `model(x)` on an input `x` to invoke the `forward` method. 

### The `nn.Parameter` class

A `nn.Parameter` is a special type of `Tensor`. Basically, this is the class that torch has provided for storing the weights and biases of a `Module`. It has some special properties for doing this:

- If a `Parameter` is set as an attribute of a `Module`, it will be auto-detected by torch and returned when you call `module.parameters()` (along with all the other `Parameters` associated with the `Module`, or any of the `Module`'s sub-modules!).
- This makes it easy to pass all the parameters of a model into an optimizer and update them all at once.

When you create a `Module` that has weights or biases, be sure to wrap them in `nn.Parameter` so that torch can detect and update them appropriately:

```python
def __init__(self, weights: t.Tensor, biases: t.Tensor):
    super().__init__()
    self.weights = nn.Parameter(weights) # wrapping a tensor in nn.Parameter
    self.biases = nn.Parameter(biases)
```

### Printing information with `extra_repr`

Although the code above covers all the essential parts of creating a module, we will add one more method: `extra_repr`. This sets the extra representation of a module - in other words, if you have a module `class MyModule(nn.Module)`, then when you print an instance of this module, it will return the (formatted) string `f"MyModule({extra_repr})"`. You might want to take this opportunity to print out useful invariant information about the module (e.g. `kernel_size`, `stride` or `padding`). The Python built-in function `getattr` might be helpful here (it can be used e.g. as `getattr(self, "padding")`, which returns the same as `self.padding` would).

## Exercise 1 - `MaxPool2d` module

```python
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        pass

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        pass

utils.test_maxpool2d_module(MaxPool2d)
m = MaxPool2d(kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")
```
""")

    with st.expander("""Help - I'm really confused about what to do here!"""):
        st.markdown("""Your `forward` method should just implement the `maxpool2d` function that you defined earlier in these exercises. In order to get the parameters for this function like `kernel_size` and `stride`, you'll need to initialise them in `__init__`. 

Later modules will be a bit more complicated because you'll need to initialise weights, but `MaxPool2d` has no weights - it's just a wrapper for the `maxpool2d` function.

---

You want the `extra_repr` method to output something like:
```python
"kernel_size=3, stride=2, padding=1"
```

so that when you print the module, it will look like this:

```python
MaxPool2d(kernel_size=3, stride=2, padding=1)
```
""")

    with st.expander("Help - I get the error 'MaxPool2d' object has no attribute '_backward_hooks'"):
        st.markdown("""Remember to call `super().__init__()` in all your `Module` subclasses. This is a very easy thing to forget!""")

    st.markdown("""
## Exercise 2 - `ReLU` and `Flatten`

Now, you should do the same for the functions `ReLU` and `Flatten`. Neither of these have learnable parameters, so they should both follow exactly the same pattern as `MaxPool2d` above. Make sure you look at the PyTorch documentation pages for [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) and [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) so that you're comfortable with what they do and why they're useful in neural networks.""")

    with st.expander("""Question - in a CNN, should you have Flatten layers before or after convolutional layers?"""):
        st.markdown("""Flatten is most often used to stack over all non-batched dimensions, which includes the height and width dimensions of an image. This will destroy spatial relationships, meaning you should do it **after** you've done all your convolutions.
    
`Flatten` is usually only used after convolutions, before applying fully connected linear layers. For an example of this, see the CNN and ResNet architectures in tomorrow's exercise which we'll be attempting to build.""")

    st.markdown("""Note that ReLU's constructor has no arguments, so it doesn't need an `extra_repr`.

```python
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        pass

utils.test_relu(ReLU)
```

```python
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        pass

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        pass

    def extra_repr(self) -> str:
        pass

utils.test_flatten(Flatten)
```
""")

    with st.expander("""Help - I'm not sure which function to use for Flatten."""):
        st.markdown("""You could use `einops.rearrange`, but constructing the rearrangement pattern as a string is nontrivial. Using `torch.reshape` will be easier.""")

    with st.expander("""Help - I can't figure out what shape the output should be in Flatten."""):
        st.markdown("""
If `input.shape = (n0, n1, ..., nk)`, and the `Flatten` module has `start_dim=i, end_dim=j`, then the new shape should be `(n0, n1, ..., ni*...*nj, ..., nk)`. This is because we're **flattening** over these dimensions.

Try first constructing this new shape object (you may find `functools.reduce` helpful for taking the product of a list), then using `torch.reshape` to get your output.
""")

    with st.expander("""Help - I can't see why my Flatten module is failing the tests."""):
        st.markdown("""
The most common reason is failing to correctly handle indices. Make sure that:
* You're indexing up to **and including** `end_dim`.
* You're correctly managing the times when `end_dim` is negative (e.g. if `input` is an nD tensor, and `end_dim=-1`, this should be interpreted as `end_dim=n-1`).
""")

    st.markdown("""
## Exercise 3 - `Linear` module

Now implement your own `Linear` module. This applies a simple linear transformation, with a weight matrix and optional bias vector. The PyTorch documentation page is [here](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html). Note that this is the first `Module` you'll implement that has learnable weights and biases.""")

    with st.expander("""Question - what type should these variables be?"""):
        st.markdown("""They have to be `torch.Tensor` objects wrapped in `nn.Parameter` in order for `nn.Module` to recognize them. If you forget to do this, `module.parameters()` won't include your `Parameter`, which prevents an optimizer from being able to modify it during training. 
        
Also, in tomorrow's exercises we'll be building a ResNet and loading in weights from a pretrained model, and this is hard to do if you haven't registered all your parameters!""")

    st.markdown(r"""For any layer, initialization is very important for the stability of training: with a bad initialization, your model will take much longer to converge or may completely fail to learn anything. The default PyTorch behavior isn't necessarily optimal and you can often improve performance by using something more custom, but we'll follow it for today because it's simple and works decently well.

Each float in the weight and bias tensors are drawn independently from the uniform distribution on the interval:

$$ \bigg[-\frac{1}{\sqrt{N_{in}}}, \frac{1}{\sqrt{N_{in}}}\bigg] $$

where $N_{in}$ is the number of inputs contributing to each output value. The rough intuition for this is that it keeps the variance of the activations at each layer constant, since each one is calculated by taking the sum over $N_{in}$ inputs multiplied by the weights (and standard deviation of the sum of independent random variables scales as the square root of number of variables).

The name for this is **Xavier (uniform) initialisation**.""")

    st.markdown("""
```python
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        pass

    def extra_repr(self) -> str:
        pass
    
utils.test_linear_forward(Linear)
utils.test_linear_parameters(Linear)
utils.test_linear_no_bias(Linear)
```""")

    with st.expander("""Help - when I print my Linear module, it also prints a large tensor."""):
        st.markdown("""This is because you've (correctly) defined `self.bias` as either `torch.Tensor` or `None`, rather than set it to the boolean value of `bias` used in initialisation.
        
To fix this, you will need to change `extra_repr` so that it prints the boolean value of `bias` rather than the value of `self.bias`.""")

    st.markdown("""## Exercise 4 - `Conv2d` module

Finally, we'll implement a module version of our `conv2d` function. This should look very similar to our linear layer implementation above, with weights and an optional bias tensor. The `nn.Conv2d` documentation page can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html). You should implement this without any bias term.
    """)

    with st.expander("""Help - I don't know what to use as number of inputs, when doing Xavier initialisation."""):
        st.markdown("""In the case of convolutions, each value in the output is computed by taking the product over `in_channels * kernel_width * kernel_height` elements. So this should be our value for $N_{in}$.""")

    st.markdown("""
```python
class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        pass

    def extra_repr(self) -> str:
        pass

utils.test_conv2d_module(Conv2d)
```
---

Congratulations for getting to the end of day 2! That was a lot of material we covered!""")

    st.button("Press me when you're finished! 🙂", on_click = st.balloons)

    st.markdown("""We'll start day 3 by using all of our code from the last section to build a CNN, and construct a basic training loop for our neural network.

We'll then proceed to a more advanced architecture: residual neural networks, to classify ImageNet images.

We'd be grateful if you could give feedback on today's exercises - you can find the form [here](https://forms.gle/cEt3zzWyPcM2Cgrr5).
""")


func_list = [section0, section1, section2, section3, section4]

page_list = ["🏠 Home", "1️⃣ Einops and Einsum", "2️⃣ Array strides", "3️⃣ Convolutions", "4️⃣ Making your own modules"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()
