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
fig_dict = st.session_state["fig_dict"]

st.set_page_config(layout="wide")

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

def section_home():
    st.markdown("""
## Build Your Own Backpropagation Framework

Today you're going to build your very own system that can run the backpropagation algorithm in essentially the same way as PyTorch does. By the end of the day, you'll be able to train a multi-layer perceptron neural network, using your own backprop system!

The main differences between the full PyTorch and our version are:

* We will focus on CPU only, as all the ideas are the same on GPU.
* We will use NumPy arrays internally instead of ATen, the C++ array type used by PyTorch. Backpropagation works independently of the array type.
* A real torch.Tensor has about 700 fields and methods. We will only implement a subset that are particularly instructional and/or necessary to train the MLP.

Note - for today, I'd lean a lot more towards being willing to read the solutions, and even move on from some of them if you don't fully understand them. The low-level messy implementation details for today are much less important than the high-level conceptual takeaways.

Also, if you don't have enough time to finish all sections (which is understandable, because there's a *lot* of content today!), I'd focus on sections **1️⃣ Introduction** and **2️⃣ Autograd**, since conceptually these are the most important. Once you've done both of these, you should have a strong working understanding of the mechanics of backpropagation.

## 1️⃣ Introduction

This takes you through what a **computational graph** is, and the basics of how gradients can be backpropagated through such a graph. You'll also implement the backwards versions of some basic functions: if we have tensors `output = func(input)`, then the backward function of `func` can calculate the grad of `input` as a function of the grad of `output`.

## 2️⃣ Autograd

This section goes into more detail on the backpropagation methodology. In order to find the `grad` of each tensor in a computational graph, we first have to perform a **topological sort** of the tensors in the graph, so that each time we try to calculate `tensor.grad`, we've already computed all the other gradients which are used in this calculation. We end this section by writing a `backprop` function, which works just like the `tensor.backward()` method you're already used to in PyTorch.

## 3️⃣ More forward & backward functions

Now that we've established the basics, this section allows you to add more forward and backward functions, extending the set of functions we can use in our computational graph.

## 4️⃣ Putting everything together

In this section, we build your own equivalents of `torch.nn` features like `nn.Parameter`, `nn.Module`, and `nn.Linear`. We can then use these to build our own neural network to classify MINST data.

This completes the chain which starts at basic numpy arrays, and ends with us being able to build essentially any neural network architecture we want!

## 5️⃣ Bonus

A few bonus exercises are suggested, for pushing your understanding of backpropagation further.
""")

def section_intro():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#reading">Reading</a></li>
    <li><a class="contents-el" href="#imports">Imports</a></li>
    <li><a class="contents-el" href="#computing-gradients-with-backpropagation">Computing Gradients with Backpropagation</a></li>
    <li><a class="contents-el" href="#transforms">Transforms</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#backward-functions">Backward Functions</a></li>
        <li><a class="contents-el" href="#topological-ordering">Topological Ordering</a></li>
        <li><a class="contents-el" href="#backpropagation">Backpropagation</a></li>
    </li></ul>
    <li><a class="contents-el" href="#backward-function-of-log">Backward function of log</a></li>
    <li><a class="contents-el" href="#backward-functions-of-two-tensors">Backward functions of two tensors</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#broadcasting-rules">Broadcasting Rules</a></li>
        <li><a class="contents-el" href="#backward-function-for-elementwise-multiply">Backward Function for Elementwise Multiply</a></li>
    </li></ul>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""
## Reading

* [Calculus on Computational Graphs: Backpropagation (Chris Olah)](https://colah.github.io/posts/2015-08-Backprop/)

## Imports

```python
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol, Union
from einops import repeat
import utils

Arr = np.ndarray
grad_tracking_enabled = True
```

## Computing Gradients with Backpropagation

This section will briefly review the backpropagation algorithm, but focus mainly on the concrete implementation in software.

To train a neural network, we want to know how the loss would change if we slightly adjust one of the learnable parameters.

One obvious and straightforward way to do this would be just to add a small value  to the parameter, and run the forward pass again. This is called finite differences, and the main issue is we need to run a forward pass for every single parameter that we want to adjust. This method is infeasible for large networks, but it's important to know as a way of sanity checking other methods.

A second obvious way is to write out the function for the entire network, and then symbolically take the gradient to obtain a symbolic expression for the gradient. This also works and is another thing to check against, but the expression gets quite complicated.

Suppose that you have some **computational graph**, and you want to determine the derivative of the some scalar loss L with respect to NumPy arrays a, b, and c:""")

    st.write("""<figure style="max-width:550px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNpNjzEOgzAMRa8SeegEA4wZKlVqN7q0axaDTUEigNJkqCLuXpMCqocv67832BGaiRg0vBzOnaoeZlQyqE6qVnl-VvcwFFFi-YFaQJPAhaiIEhtYvdTTTss_usqp54MeoExVBRlYdhZ7kmviSg34ji0b0LIStxgGb8CMi6hhJvR8o95PDnSLw5szwOCn52dsQHsXeJeuPcpzdrOWLxMNSCM" /></figure>""", unsafe_allow_html=True)

    st.markdown(r"""This graph corresponds to the following Python:

```python
d = a * b
e = b + c
L = d + e
```

The goal of our system is that users can write ordinary looking Python code like this and have all the book-keeping needed to perform backpropagation happen behind the scenes. To do this, we're going to wrap each array and each function in special objects from our library that do the usual thing plus build up this graph structure that we need.

### Backward Functions

We've drawn our computation graph from left to right and the arrows pointing to the right, so that in the forward pass, boxes to the right depend on boxes to the left. In the backwards pass, the opposite is true: the gradient of boxes on the left depends on the gradient of boxes on the right.

If we want to compute the derivative of $L$ wrt all other variables (as was described in the reading), we should traverse the graph from right to left. Each time we encounter an instance of function application, we can use the chain rule from calculus to proceed one step further to the left. For example, if we have $d = a \times b$, then:

$$\frac{dL}{da} = \frac{dL}{dd}\times \frac{dd}{da} = \frac{dL}{dd}\times b$$

Suppose we are working from right to left, trying to calculate $\frac{dL}{da}$. If we already know the values of the variables $a$, $b$ and $d$, as well as the value of $\frac{dL}{dd}$, then we can use the following function to find $\frac{dL}{da}$:

$$F(a, b, d, \frac{\partial L}{\partial d}) = \frac{\partial L}{\partial d}\cdot b$$

and we can do something similar when trying to calculate $\frac{dL}{db}$.

In other words, we can take the "forward function" $(a, b) \to a \cdot b$, and for each of its parameters, we can define an associated "backwards function" which tells us how to compute the gradient wrt this argument using only known quantities as inputs.

### Topological Ordering

When we're actually doing backprop, how do we guarantee that we'll always know the value of our backwards functions' inputs? For instance, in the example above we couldn't have computed $\frac{dL}{da}$ without first knowing $\frac{dL}{dd}$.

The answer is that we sort all our nodes using an algorithm called [topological sorting](https://en.wikipedia.org/wiki/Topological_sorting), and then do our computations in this order. After each computation, we store the gradients in our nodes for use in subsequent calculations.

When described in terms of the diagram above, topological sort can be thought of as an ordering of nodes from right to left. Crucially, this sorting has the following property: if there is a directed path in the computational graph going from node `x` to node `y`, then `x` must follow `y` in the sorting. 

There are many ways of proving that a cycle-free directed graph contains a topological ordering. You can try and prove this for yourself, or click on the expander below to reveal the outline of a simple proof.
""")

    with st.expander("Click to reveal"):
        st.markdown("""We can prove by induction on the number of nodes $N$. 
    
If $N=1$, the problem is trivial.

If $N>1$, then pick any node, and follow the arrows until you reach a node with no directed arrows going out of it. Such a node must exist, or else you would be following the arrows forever, and you'd eventually return to a node you previously visited, but this would be a cycle, which is a contradiction. Once you've found this "root node", you can put it first in your topological ordering, then remove it from the graph and apply the topological sort on the subgraph created by removing this node. By induction, your topological sorting algorithm on this smaller graph should return a valid ordering. If you append the root node to the start of this ordering, you have a topological ordering for the whole graph.

Induction complete.""")

    st.markdown("""
A quick note on some potentially confusing terminology. In some contexts (e.g. causal inference), it's common to call nodes with no arrows coming out of them "root nodes", and nodes with no arrows going into them "leaf nodes" (so in the diagram above, the left nodes would be roots and the right nodes would be leaves).

When we talk about computational graphs, the language is the other way around. In the diagram above, `a`, `b` and `c` are the leaf nodes, and `L` is the root node. 

Another important piece of terminology here is **parent node**. This means the same thing as it does in most other contexts - the parents of node `x` are all the nodes `y` with connections `y -> x` (so in the diagram, `d`'s parents are `a` and `b`).
""")

    with st.expander("Question - can you think of a reason it might be important for a node to store a list of all of its parent nodes?"):
        st.markdown("""During backprop, we're moving from right to left in the diagram. If a node doesn't store its parent, then there will be no way to get access to that parent node during backprop, so we can't propagate gradients to it.""")

    st.markdown(r"""The very first node in our topological sort will be $L$, the root node.

### Backpropagation

After all this setup, the backpropagation mechanism becomes pretty straightforward. We sort the nodes topologically, then we iterate over them and call each backward function exactly once in order to accumulate the gradients at each node.

It's important that the grads be accumulated instead of overwritten in a case like value $b$ which has two outgoing edges, since $dL/db$ will then be the sum of two terms. Since addition is commutative it doesn't matter whether we `backward()` the Mul or the Add that depend on $b$ first.

During backpropagation, for each forward function in our computational graph we need to find the partial derivative of the output with respect to each of its inputs. Each partial is then multiplied by the gradient of the loss with respect to the forward functions output (`grad_out`) to find the gradient of the loss with respect to each input. We'll handle these calculations using backward functions.

## Backward function of log

First, we'll write the backward function for `x -> out = log(x)`. This should be a function which, when fed the values `x, out, dL/d(out)` returns the value of `dL/dx` based on the contribution from this particular computational path.""")

    st.write("""<figure style="max-width:400px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNo1zbEOwjAMBNBfiW5ufyADE2ydYM1iNW4bqYmr4Eigqv9eg-CWu-FJt2OUyPCYK22LG-6hOMvL9f3FDTJ_W5qiQ-aaKUWz-wcF6MKZA7zNyBO1VQNCOYy2LZLyLSaVCj_R-uQO1FQe7zLCa238R9dEdp1_6jgBox4vQQ" /></figure>""", unsafe_allow_html=True)


    st.markdown(r"""
Note - it might seem strange at first why we need `x` and `out` to be inputs, `out` can be calculated directly from `x`. The answer is that sometimes it is computationally cheaper to express the derivative in terms of `out` than in terms of `x`.""")

    with st.expander("""Question - can you think of an example function where it would be computationally cheaper to use 'out'?"""):
        st.markdown("""The most obvious answer is the exponential function, `out = e ^ x`. Here, the gradient `d(out)/dx` is equal to `out`. We'll see this when we implement a backward version of `torch.exp` later today.""")

    with st.expander("""Question - what should the output of this backward function for log be?"""):
        st.markdown(r"""By the chain rule, we have:
$$
\frac{dL}{dx} = \frac{dL}{d(\text{out})} \cdot \frac{d(\text{out})}{dx} = \frac{dL}{d(\text{out})} \cdot \frac{d(\log{x})}{dx} = \frac{dL}{d(\text{out})} \cdot \frac{1}{x}
$$""")

    st.markdown("""
You should now fill in this function below. Also don't worry about division by zero or other edge cases - the goal here is just to see how the pieces of the system fit together.

```python
def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backwards function for f(x) = log(x)

    grad_out: gradient of some loss wrt out
    out: the output of np.log(x)
    x: the input of np.log

    Return: gradient of the given loss wrt x
    '''
    pass

utils.test_log_back(log_back)
```
## Backward functions of two tensors

Now we'll implement backward functions for multiple tensors. To do so, we first need to understand broadcasting.

### Broadcasting Rules

Both NumPy and PyTorch have the same rules for broadcasting. The shape of the arrays being operated on is compared element-wise, starting from the rightmost dimension and working left. Two dimensions are compatible when

* they are equal, or
* one of them is 1 (in which case the array is repeated along this dimension to fit into the other one).

Two arrays with a different number of dimensions can be operated on, provided the one with fewer dimensions is compatible with the rightmost elements of the one with more dimensions. Another way to picture this is that NumPy appends dimensions of size 1 to the start of the smaller-dimensional array until they both have the same dimensionality, and then their sizes are checked for compatibility.

As a warm-up exercise, below are some examples of broadcasting. Can you figure out which are valid, and which will raise errors?

```python
x = np.ones((3, 1, 5))
y = np.ones((1, 4, 5))

z = x + y
```
""")

    with st.expander("""Answer"""):
        st.markdown("""This is valid, because the 0th dimension of `y` and the 1st dimension of `x` can both be copied so that `x` and `y` have the same shape: `(3, 4, 5)`. The resulting array `z` will also have shape `(3, 4, 5)`.""")

    st.markdown("""
```python
x = np.ones((8, 2, 6))
y = np.ones((8, 2))

z = x + y
```""")

    with st.expander("""Answer"""):
        st.markdown("""This is not valid. We first need to expand `y` by appending a dimension to the front, and the last two dimensions of `x` are `(2, 6)`, which won't broadcast with `y`'s `(8, 2)`.""")

    st.markdown("""
```python
x = np.ones((8, 2, 6))
y = np.ones((2, 6))

z = x + y
```""")

    with st.expander("""Answer"""):
        st.markdown("""This is valid. Once NumPy expands `y` by appending a single dimension to the front, it can then be broadcast with `x`.""")


# with st.expander("""Help - I'm not sure what this function is asking for."""):
#     st.markdown(r"""We effectively have $\frac{dL}{d(\text{out})}$, $\text{out}$ and $x$ as inputs to our `log_back` function, and we're trying to compute $\frac{dL}{dx}$. The answer was provided in the expander above.""")

    st.markdown("""
When writing our backward functions, we'll need to deal with broadcasting. For instance, suppose we have the operation `c = f(a + b)` where `a.shape = (3, 4)` and `b.shape = (4,)`. In this case, `b` is broadcasted in order to be added to `a`, and so if we want to compute the derivative of `c` wrt `b`, we first need to compute the derivative wrt the broadcasted version of `b`, then sum over the 0th dimension to get the true derivative. 

This is because each element `b[j]` appears not once, but three times in the function `f`. For instance, if `f` was just the function `.sum()` then each `b[j]` would be summed three times rather than once. **We want the grad with respect to an input element to be the sum of each position where that input element was used in the output.**

In our backward functions, we'll need to implement sum form of this "reverse broadcasting", where we sum over the dimensions that were expanded during broadcasting. The way we'll implement this is via the `unbroadcast` helper function below.
""")

    with st.expander("""Help - I'm confused about implementing unbroadcast!"""):
        st.markdown("""`unbroadcast` has to do two things:
1. Sum and remove dimensions that were prepended to the front of the original shape.
2. Sum dimensions that were originally 1 back to the size 1 (using `keepdims=True`).
It's easiest to do the two tasks in that order, since after removing the additional dimensions, the shapes will have an equal number of dimensions.""")

    st.markdown("""
```python
def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    '''Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    '''
    pass

utils.test_unbroadcast(unbroadcast)
```

### Backward Function for Elementwise Multiply

Functions that are differentiable with respect to more than one input tensor are straightforward given that we already know how to handle broadcasting.

- We're going to have two backwards functions, one for each input argument.
- If the input arguments were broadcasted together to create a larger output, the incoming `grad_out` will be of the larger common broadcasted shape and we need to make use of `unbroadcast` from earlier to match the shape to the appropriate input argument.
- We'll want our backward function to work when one of the inputs is an float. We won't need to calculate the grad_in with respect to floats, so we only need to consider when y is an float for `multiply_back0` and when x is an float for `multiply_back1`.

```python
def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Union[Arr, float]) -> Arr:
    '''Backwards function for x * y wrt argument 0 aka x.'''
    pass

def multiply_back1(grad_out: Arr, out: Arr, x: Union[Arr, float], y: Arr) -> Arr:
    '''Backwards function for x * y wrt argument 1 aka y.'''
    pass

utils.test_multiply_back(multiply_back0, multiply_back1)
utils.test_multiply_back_float(multiply_back0, multiply_back1)
```

You might be wondering why we need a `multiply_back0` and a `multiply_back1` function, rather than just having a single function to serve both purposes. This will become more important later on, once we deal with functions with more than one argument, which is not symmetric in its arguments. For instance, the derivative of $x / y$ wrt $x$ is not the same as the expression you get after differentiating this wrt $y$ then swapping the labels around.

Now we'll use our backward functions to do backpropagation manually, for the following computational graph:""")

    st.write("""<figure style="max-width:800px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNpFj7EOgzAMRH8l8ljBAGOGTu3G1I6kgxubgEQSlCZDhfj3plCKB8s-PZ_OM2hPDBKUMwGnXjQ35UQuFGV5FlPwVLWnx7rQX6p3qVv76E3VNt5smtkMnofBJuidrQ-WD0vloADLweJAOc_8PVIQe7asQOaRuMM0RpWjLhlNE2HkKw3RB5Adji8uAFP097fTIGNIvEOXAfNz9kctHyZgSew" /></figure>""", unsafe_allow_html=True)

    st.markdown("""
This should give you a feel for how the backwards functions that you've defined so far will actually work in practice (although eventually you'll be automating these steps in a way similar to PyTorch's `autograd`).



```python
def forward_and_back(a: Arr, b: Arr, c: Arr) -> tuple[Arr, Arr, Arr]:
    '''
    Calculates the output of the computational graph above (g), then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    '''
    d = a * b
    e = np.log(c)
    f = d * e
    g = np.log(f)

    final_grad_out = np.array([1.0])

    # Your code here

utils.test_forward_and_back(forward_and_back)
```
""")

    with st.expander("""Help - I'm not sure how to start!"""):
        st.markdown("""The first line you add inside your function should be something like:

```python
dg_df = log_back(grad_out=np.ones(1), out=g, x=f)
```

because `f` is the input to the `log` function, `g` is the output, and we use 1 as `grad_out` because we're starting from the root node.

You should work backwards in this way, until you get to `dg_da, dg_db, dg_dc`. The gradients from previous computations should be the `grad_out` arguments of subsequent computations.""")

def section_autograd():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#wrapping-arrays-tensor">Wrapping Arrays (Tensor)</a></li>
    <li><a class="contents-el" href="#recipe">Recipe</a></li>
    <li><a class="contents-el" href="#registering-backwards-functions">Registering backwards functions</a></li>
    <li><a class="contents-el" href="#tensors">Tensors</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#requires-grad"><code>requires_grad</code></a></li>
    </li></ul>
    <li><a class="contents-el" href="#forward-pass-building-the-computational-graph">Forward Pass: Building the Computational Graph
    <li><a class="contents-el" href="#forward-pass-generic-version">Forward Pass: Generic Version</a></li>
    <li><a class="contents-el" href="#backpropagation">Backpropagation</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#topological-sort">Topological Sort</a></li>
        <li><a class="contents-el" href="#the-backward-method"></a>The <code>backward</code> method</li>
        <li><a class="contents-el" href="#end-grad">End grad</a></li>
        <li><a class="contents-el" href="#the-backprop-function">The <code>backprop</code> function</a></li>
    </li></ul>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Autograd
Now, rather than figuring out which backward functions to call, in what order, and what their inputs should be, we'll write code that takes care of that for us. We'll implement this with a few major components:
- Tensor
- Recipe
- wrap_forward_fn

## Wrapping Arrays (Tensor)
We're going to wrap each array with a wrapper object from our library which we'll call `Tensor` because it's going to behave similarly to a `torch.Tensor`.

Each Tensor that is created by one of our forward functions will have a `Recipe`, which tracks the extra information need to run backpropagation.

`wrap_forward_fn` will take a forward function and return a new forward function that does the same thing while recording the info we need to do backprop in the `Recipe`.

## Recipe
Let's start by taking a look at `Recipe`.

`@dataclass` is a handy class decorator that sets up an `__init__` function for the class that takes the provided attributes as arguments and sets them as you'd expect.

The class `Recipe` is designed to track the forward functions in our computational graph, so that gradients can be calculated during backprop. Each tensor created by a forward function has its own `Recipe`. We're naming it this because it is a set of instructions that tell us which ingredients went into making our tensor: what the function was, and what tensors were used as input to the function to produce this one as output.

```python
@dataclass(frozen=True)
class Recipe:
    '''Extra information necessary to run backpropagation. You don't need to modify this.'''

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll create for it later on."
    args: tuple
    "The input arguments passed to func."
    "For instance, if func was np.sum then args would be a length-1 tuple containing the tensor to be summed."
    kwargs: dict[str, Any]
    "Keyword arguments passed to func."
    "For instance, if func was np.sum then kwargs might contain 'dim' and 'keepdims'."
    parents: dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position, in order to be able to pass gradients back along the computational graph."
```

As an example, if a tensor `y` was created via `np.func(x0.array, x1.array, ..., xN.array)`, then the recipe for `y` might look like this:

```python
y.recipe = Recipe(func=np.func, args=(x0.array, x1.array, ..., xN.array), kwargs={}, parents={0: x0, 1: x1, ..., N: xN})
```

Note that `args` just stores the values of the underlying arrays, but `parents` stores the actual tensors. This is because they serve two different purposes: `args` is required for computing the value of gradients during backpropagation, and `parents` is required to infer the structure of the computational graph (i.e. which tensors were used to produce which other tensors).

## Registering backwards functions

The `Recipe` takes care of tracking the forward functions in our computational graph, but we still need a way to find the backward function corresponding to a given forward function when we do backprop (or possibly the set of backward functions, if the forward function takes more than one argument). We will do this by defining a class `BackwardFuncLookup`. Details of the implementation are left up to you.

The implementation today can be done very simply. We won't support backprop wrt keyword arguments and will raise an exception if the user tries to pass a Tensor by keyword. You can remove this limitation later if you have time.

We do need to support functions with multiple positional arguments like multiplication so we'll also provide the positional argument index when setting and getting back_fns.

```python
class BackwardFuncLookup:
    def __init__(self) -> None:
        pass

    def add_back_func(self, forward_fn: Callable, arg_position: int, back_fn: Callable) -> None:
        pass

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        pass

utils.test_back_func_lookup(BackwardFuncLookup)

BACK_FUNCS = BackwardFuncLookup()
BACK_FUNCS.add_back_func(np.log, 0, log_back)
BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)
```
""")
    with st.expander("Help - I'm not sure how to implement BackwardFuncLookup."):
        st.markdown(r"""
Try defining a lookup dictionary `self.lookup = {}` in the `__init__` stage. This should map forward functions to their corresponding backward functions.
""")

    st.markdown("""
## Tensors
Our Tensor object has these fields:
- An `array` field of type `np.ndarray`.
- A `requires_grad` field of type `bool`.
- A `grad` field of the same size and type as the value.
- A `recipe` field, as we've already seen.

### requires_grad
The meaning of `requires_grad` is that when doing operations using this tensor, the recipe will be stored and it and any descendents will be included in the computational graph.

Note that `requires_grad` does not mean that we will save the accumulated gradients to this tensor's `.grad` parameter when doing backprop: we will follow pytorch's implementation of backprop and only save gradients to leaf tensors (see `Tensor.is_leaf`, below).

There is a lot of repetitive boilerplate involved which we have done for you. You don't need to modify anything in this class: the methods here will delegate to functions that you will implement throughout the day. You should read the code for the `Tensor` class up to `__init__`, and make sure you understand it. Most of the methods beyond this are just replicating the basic functionality of PyTorch tensors.

##### There's a lot of code in the block below, which is why I've put it in an expander (otherwise it's a pain to scroll past!). You should copy this code into your `answers.py` file and run it.
""")
    with st.expander("CODE TO RUN"):
        st.markdown("""
```python
Arr = np.ndarray
from typing import Optional, Union

class Tensor:
    '''
    A drop-in replacement for torch.Tensor supporting a subset of features.
    '''

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
        '''Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html'''
        if self.requires_grad and self.recipe and self.recipe.parents:
            return False
        return True

    def __bool__(self):
        if np.array(self.shape).prod() != 1:
            raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
        return bool(self.item())

def empty(*shape: int) -> Tensor:
    '''Like torch.empty.'''
    return Tensor(np.empty(shape))

def zeros(*shape: int) -> Tensor:
    '''Like torch.zeros.'''
    return Tensor(np.zeros(shape))

def arange(start: int, end: int, step=1) -> Tensor:
    '''Like torch.arange(start, end).'''
    return Tensor(np.arange(start, end, step=step))

def tensor(array: Arr, requires_grad=False) -> Tensor:
    '''Like torch.tensor.'''
    return Tensor(array, requires_grad=requires_grad)
```""")

    st.markdown("""
## Forward Pass: Building the Computational Graph

Let's start with a simple case: our `log` function. `log_forward` is a wrapper, which should implement the functionality of `np.log` but work with tensors rather than arrays. Our `log` function must do the following:

- Call `np.log(x.array)` to obtain an output array.
- Create a new `Tensor` containing the output.
- If grad tracking is enabled globally AND (the input requires grad, OR has a recipe), then the output requires grad and we fill out the recipe of our output, as a `Recipe` object.
 
Later we'll redo this in a generic and reusable way, but for now just get it working.

```python
def log_forward(x: Tensor) -> Tensor:
    pass

log = log_forward
test_log(Tensor, log_forward)
test_log_no_grad(Tensor, log_forward)
a = Tensor([1], requires_grad=True)
grad_tracking_enabled = False
b = log_forward(a)
grad_tracking_enabled = True
assert not b.requires_grad, "should not require grad if grad tracking globally disabled"
assert b.recipe is None, "should not create recipe if grad tracking globally disabled"
```""")

    with st.expander("Help - I don't know what the recipe object should be."):
        st.markdown("""
Your recipe should look like this:

```python
Recipe(func=np.log, args=(x.array,), kwargs={}, parents={0: x})
```

Explanation: you're working with the function `np.log`, which just has a single argument (the array that you're taking the log of), and has no keyword arguments.
""")

    st.markdown("""
Now let's do the same for multiply, to see how to handle functions with multiple arguments. There are a few differences:
- The actual function to be called is different
- We need more than one argument in `args` and `parents`, when defining `Recipe`
- `requires_grad` should be true if `grad_tracking_enabled=True`, and ANY of the input tensors require grad
- One of the inputs may be an int, so you'll need to deal with this case before calculating `out`

```python
def multiply_forward(a: Union[Tensor, int], b: Union[Tensor, int]) -> Tensor:
    pass

multiply = multiply_forward
utils.test_multiply(Tensor, multiply_forward)
utils.test_multiply_no_grad(Tensor, multiply_forward)
utils.test_multiply_float(Tensor, multiply_forward)
a = Tensor([2], requires_grad=True)
b = Tensor([3], requires_grad=True)
grad_tracking_enabled = False
b = multiply_forward(a, b)
grad_tracking_enabled = True
assert not b.requires_grad, "should not require grad if grad tracking globally disabled"
assert b.recipe is None, "should not create recipe if grad tracking globally disabled"
```
""")

    with st.expander("""Help - I get "AttributeError: 'int' object has no attribute 'array'"."""):
        st.markdown("""Remember that your multiply function should also accept integers. You need to separately deal with the cases where `a` and `b` are integers or Tensors.""")
    
    with st.expander("""Help - I don't know what should go into 'parents' in the recipe."""):
        st.markdown("""Remember that `parents` is only used for inferring the structure of the computational graph. You only need to store the arguments `a` and `b` if they are tensors, *not if they are integers*.""")

    st.markdown("""
## Forward Pass - Generic Version
All our forward functions are going to look extremely similar to `log_forward` and `multiply_forward`. 
Implement the higher order function `wrap_forward_fn` that takes a `Arr -> Arr` function and returns a `Tensor -> Tensor` function. In other words, `wrap_forward_fn(np.log)` should evaluate to a callable that does the same thing as your `log_forward`.""")

    st.markdown("""
```python
def wrap_forward_fn(numpy_func: Callable, is_differentiable=True) -> Callable:
    '''
    numpy_func: function. It takes any number of positional arguments, some of which may be NumPy arrays, and any number of keyword arguments which we aren't allowing to be NumPy arrays at present. It returns a single NumPy array.
    is_differentiable: if True, numpy_func is differentiable with respect to some input argument, so we may need to track information in a Recipe. If False, we definitely don't need to track information.

    Return: function. It has the same signature as numpy_func, except wherever there was a NumPy array, this has a Tensor instead.
    '''

    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        pass

    return tensor_func

log = wrap_forward_fn(np.log)
multiply = wrap_forward_fn(np.multiply)
# need to be careful with sum, because kwargs have different names in torch and numpy
def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    return np.sum(x, axis=dim, keepdims=keepdim)
sum = wrap_forward_fn(_sum)

utils.test_log(Tensor, log)
utils.test_log_no_grad(Tensor, log)
utils.test_multiply(Tensor, multiply)
utils.test_multiply_no_grad(Tensor, multiply)
utils.test_multiply_float(Tensor, multiply)
utils.test_sum(Tensor)
try:
    log(x=Tensor([100]))
except Exception as e:
    print("\nGot a nice exception as intended:")
    print(e)
else:
    assert False, "Passing tensor by keyword should raise some informative exception."
```
""")

    with st.expander("""Help - I'm not sure where to start."""):
        st.markdown("""This will be structured very similarly to the `multiply_forward` function (you can look at the solution if you're still unsure about your implementation of this). The main difference is that you're iterating over args, rather than knowing that there are two.""")

    with st.expander("""Help - I'm getting "NameError: name 'getitem' is not defined"."""):
        st.markdown("""This is probably because you're calling `numpy_func` on the args themselves. Recall that `args` will be a list of `Tensor` objects, and that you should call `numpy_func` on the underlying arrays.""")

    with st.expander("""Help - I'm getting an AssertionError on "assert c.requires_grad == True", or something similar."""):
        st.markdown("""This is probably because you're not defining `requires_grad` correctly. Remember that the output of a forward function should have `requires_grad = True` if and only if all of the following hold:

* Grad tracking is enabled
* The function is differentiable
* **Any** of the inputs are tensors with `requires_grad = True`""")

    with st.expander("""Help - my function passes all tests up to "test_sum", but then fails here."""):
        st.markdown("""`test_sum`, unlike the previous tests, wraps a function that uses keyword arguments. So if you're failing here, it's probably because you didn't use `kwargs` correctly.

`kwargs` should be used in two ways: once when actually calling the `numpy_func`, and once when defining the `Recipe` object for the output tensor.""")

    st.markdown("""
## Backpropagation
Now all the pieces are in place to implement backpropagation. We need to:
- Loop over the nodes from right to left. At each node:
    - Call the backward function to transform the grad wrt output to the grad wrt input.
    - If the node is a leaf, write the grad to the grad field.
    - Otherwise, accumulate the grad into temporary storage.

### Topological Sort
As part of backprop, we need to sort the nodes of our graph so we can traverse the graph in the appropriate order.

Write a general function `topological_sort` that return a list of node's descendants in topological order (beginning with the furthest descentents, ending with the starting node) using [depth-first search](https://en.wikipedia.org/wiki/Topological_sorting).

We've given you a `Node` class, with a `children` attribute. You shouldn't change anything else about the `Node` class, because after this you'll be writing code to perform a topological sort on your computational graph (wherein `Node` will be replaced with the `Tensor` object).

If you're stuck, try looking at the pseudocode from some of [these examples](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm). However, it's completely fine to just look at the solution (which is an implementation of DFS) if you're stuck on this problem - this is more of a fun LeetCode-style puzzle than something intrinsic to the core of conceptually understanding backpropagation.
""")

    with st.expander("""Help - my function is hanging without returning any values."""):
        st.markdown("""This is probably because it's going around in cycles when fed a cyclic graph. You should add a way of raising an error if your function detects that the graph isn't cyclic. One way to do this is to create a set `temp`, which stores the nodes you've visited on a particular excursion into the graph, then you can raise an error if you come across an already visited node.""")

    with st.expander("Help - I'm completely stuck on how to implement this, and would like the template for some code."):
        st.markdown("""
Here is the template for a depth-first search implementation:

```python
def topological_sort(node: Node) -> list[Node]:
    
    result: list[Node] = [] # stores the list of nodes to be returned (in reverse topological order)
    temp: set[Node] = set() # keeps track of previously visited nodes (to detect cyclicity)

    def visit(cur: Node):
        '''
        Recursive function which visits all the children of the current node.

        This should update the `result` and `temp` objects (defined outside scope).
        '''
        pass # WRITE YOUR CODE HERE

    visit(node)

    return result
```
""")

    st.markdown("""
```python
class Node:
    def __init__(self, *children):
        self.children = list(children)

def topological_sort(node: Node) -> list[Any]:
    '''
    Return a list of node's descendants in reverse topological order, from future to past.
    
    Should raise an error if the graph with `node` as root is not in fact acyclic.
    '''
    pass

utils.test_topological_sort_linked_list(topological_sort)
utils.test_topological_sort_branching(topological_sort)
utils.test_topological_sort_rejoining(topological_sort)
utils.test_topological_sort_cyclic(topological_sort)
```

Now, you should write the function `sorted_computational_graph`. Just like `topological_sort`, this should return the nodes in your graph in order from future to past (i.e. in the order you'll need for backpropagation). So you can reuse much of your previous code.

Remember that the `Tensor` object stores a dictionary of its parent tensors (rather than a list of its children, like the `Node` object did).
""")

    st.markdown("""
```python
def sorted_computational_graph(node: Tensor) -> list[Tensor]:
    '''
    For a given tensor, return a list of Tensors that make up the nodes of the given Tensor's computational graph, in reverse topological order.
    '''
    pass

a = Tensor([1], requires_grad=True)
b = Tensor([2], requires_grad=True)
c = Tensor([3], requires_grad=True)
d = a * b
e = c.log()
f = d * e
g = f.log()
name_lookup = {a: "a", b: "b", c: "c", d: "d", e: "e", f: "f", g: "g"}

print([name_lookup[t] for t in sorted_computational_graph(g)])
# Should get something in reverse alphabetical order (or close)
```

### The `backward` method

Now we're really ready for backprop!

Note that in the implementation of the class `Tensor`, we had:

```python
class Tensor:

    def backward(self, end_grad: Union[Arr, "Tensor", None] = None) -> None:
        if isinstance(end_grad, Arr):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)
```

In other words, for a tensor `out`, calling `out.backward()` is equivalent to `backprop(out)`.

### End grad

You might be wondering what role the `end_grad` argument in `backward` plays. We usually just call `out.backward()` when we're working with loss functions; why do we need another argument?

The reason is that we've only ever called `out.backward()` on scalars (i.e. tensors with a single element). If `out` is multi-dimensional, then we can get a scalar from it by taking a weighted sum of all of the elements. The elements of `end_grad` are precisely the weighting coefficients that we use. In other words, calling `out.backward(end_grad)` implicitly does the following:

* Defines the value `scalar = (out * end_grad).sum()`
* Calculates the derivative of `scalar` wrt all other nodes in the graph, and uses that as their `grad` attribute

How can we implement this? We've actually already written all the code we need, when we created our backward functions. We will simply use `end_grad` as the `grad_out` argument in our backward functions, the first time we run them. For instance, consider the function `log_back` you implemented at the start, which hopefully looked something like this:

```python
def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return grad_out / x
```

Suppose we defined `x` as a 2D tensor, then `out = log(x)` is also a tensor of the same shape. Then, we run `out.backward(end_grad)`  where  `end_grad` has shape `(2, 3)`. Under the hood, the backprop function will find `x.grad` by evaluating `log_back(end_grad, out, x)`. So we get:

```python
x.grad[i, j] = end_grad[i, j] / x[i, j]
```

Now consider `scalar = (out * end_grad).sum()`. We can show mathematically that the derviative of `scalar` wrt `x[i, j]` is exactly equal to the expression above:""")

    st.markdown(r"""
$$
\begin{aligned}
\text{ scalar } &=\sum_{p q} \text { tensor }_{p q} \times \text { end\_grad}_{p q} \\
&=\sum_{p q} \log{x_{p q}} \times \text { end\_grad}_{p q} \\
\frac{\partial(\text{scalar})}{\partial x_{i j}} &=\frac{\partial}{\partial x_{i j}}\left(\,\log{x_{p q}} \times \text { end\_grad}_{i j}\right) \\
&=\text{ end\_grad}_{i j} / x_{i j}
\end{aligned}
$$
""")

    st.info("""
Mathematically, what we are doing is calculating the [directional derivative](https://en.wikipedia.org/wiki/Directional_derivative) of `out` in the direction `v`.

In other words, `x.grad[i, j]` will be the rate at which `out` changes **in the direction of `v`** as `x[i, j]` increases.
""")

    st.markdown("""
When `end_grad` is not specified, we assume it to be a tensor of 1s with the same size as `out`. For instance, if we call `L.backward()` when `L` is a scalar-valued loss function, this is the same as calling `L.backward(end_grad=Tensor(1.))`.

### Leaf nodes

The `Tensor` object has an `is_leaf` property:

```python
class Tensor:

    @property
    def is_leaf(self):
        '''Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html'''
        if self.requires_grad and self.recipe and self.recipe.parents:
            return False
        return True
```

In other words, leaf node tensors are any with either `requires_grad=False`, or which were *not* created by some operation on other tensors. 

In backprop, only the leaf nodes accumulate gradients. You can think of leaf nodes as being edges of the computational graph, i.e. nodes from which you can't move further backwards topologically. 

Intuitively, we can see that leaf nodes are the only one whose gradients we actually care about. All parameters in a neural network will be leaf nodes, because they are initialised from nothing when the network is first created. But a neural network's output will not be a leaf node, because it is the result of operations on other tensors.

In the computational graph in the next section, the only leaves are `a`, `b` and `c`.

```python
layer = torch.nn.Linear(3, 4)
input = torch.ones(3)
output = layer(input)

print(layer.weight.is_leaf) # -> True
print(output.is_leaf)       # -> False
```

### The `backprop` function

Now, we get to the actual backprop function! Some code is provided below, which you should complete.

```python
def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
    '''Accumulates gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node: 
        The rightmost node in the computation graph. 
        If it contains more than one element, end_grad must be provided.
    end_grad: 
        A tensor of the same shape as end_node. 
        If not specified, this is set to an array of 1s with same shape as end_node.array.
    '''
    # Get value of end_grad_arr
    end_grad_arr = np.ones_like(end_node.array) if end_grad is None else end_grad.array

    # Create dict to store gradients
    grads: dict[Tensor, Arr] = {end_node: end_grad_arr}

    # YOUR CODE GOES HERE

utils.test_backprop(Tensor)
utils.test_backprop_branching(Tensor)
utils.test_backprop_requires_grad_false(Tensor)
utils.test_backprop_float_arg(Tensor)
```

At a very high level, your code here should iterate through the sorted computational graph, and at each node it should do two things:

1. If the node is a leaf node, set `node.grad` from the value in the `grads` dictionary.
2. For all the node's parents, update their gradients in the `grads` dictionary using the appropriate backward function.

As an example, consider this computational graph from earlier (assume for simplicity that all nodes are scalars):""")

    st.write("""<figure style="max-width:550px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNpNjzEOgzAMRa8SeegEA4wZKlVqN7q0axaDTUEigNJkqCLuXpMCqocv67832BGaiRg0vBzOnaoeZlQyqE6qVnl-VvcwFFFi-YFaQJPAhaiIEhtYvdTTTss_usqp54MeoExVBRlYdhZ7kmviSg34ji0b0LIStxgGb8CMi6hhJvR8o95PDnSLw5szwOCn52dsQHsXeJeuPcpzdrOWLxMNSCM" /></figure>""", unsafe_allow_html=True)

    st.markdown("""
If we ran `L.backward()`, the process of running backprop would iterate through the nodes as follows:

```
(L) Update the gradients of 'd' and 'e' in the 'grads' dict
        This requires you to use the gradient of 'L' in the 'grads' dict (which is just 1), and the backward func for 'add'

(d) Update the gradients of 'a' and 'b' in the 'grads' dict
        This requires you to use the gradient of 'd' in the 'grads' dict, and the backward func for 'mul'

(e) Update the gradients of 'b' and 'c' in the 'grads' dict
        This requires you to use the gradient of 'e' in the 'grads' dict, and the backward func for 'add'

(a) Set 'a.grad' according to its value in the 'grads' dict, because 'a' is a leaf node

(b) Set 'b.grad' according to its value in the 'grads' dict, because 'b' is a leaf node

(c) Set 'c.grad' according to its value in the 'grads' dict, because 'c' is a leaf node
```

Note that in the process above, we updated `b` twice in the gradients dict. The first time, we initialise the dictionary value to be its its gradient contribution along the path `b -> d -> L`. The second time, we add its contribution along the path `b -> e -> L`. By the time we set `b.grad` from the value in the gradients dictionary, we've already added its contribution along all paths.
""")
    with st.expander("Why do we need to use the 'grads' dict at all?"):
        st.markdown("""
Technically, we could implement backprop without using the `grads` dict, by just storing gradients in nodes' `.grad` attribute. We could do this by replacing the two-step process described above with a slightly different one:

1. For all the node's parents, update `parent.grad` using the appropriate backward function.
2. If the node is *not* a leaf node, set `node.grad` to None.

This would work fine for our purposes, and the code would actually be slightly shorter. The reason we don't do it is that, as a general rule, we never want to have non-None values for non-leaf scalars. We only ever store the gradients of non-leaves in the `grads` dictionary, to avoid having to store the gradients in the leaves themselves. This is a bit annoying, but it follows the behaviour of PyTorch.
""")
    st.markdown("""
Since implementing this function will likely be very challenging, we've provided a few hints below, each one corresponding to a certain type of error you might get while trying to implement `backprop`.
""")

    with st.expander("Help - I get AttributeError: 'NoneType' object has no attribute 'func'"):
        st.markdown("""This error is probably because you're trying to access `recipe.func` from the wrong node. Possibly, you're calling your backward functions using the parents nodes' `recipe.func`, rather than the node's `recipe.func`.
        
To explain further, suppose your computational graph is simply:""")

        st.write("""<figure style="max-width:550px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNolTjEOwjAQ-0p0czvAmIGJERZYbzmSC43UJFW4DKjp37kWT7Zly17BFc9g4V1pmcztgdkoyIzjxfTQsjv11y4Ofu7GwQCJa6LotbXucQSZODGCVeo5UJsFAfOmUWpSnt_swEptPEBbPAlfI-leAhto_qjLPkqp9_-T49D2A4txMhY" /></figure>""", unsafe_allow_html=True)

        st.markdown("""When you reach `b` in your backprop iteration, you should calculate the gradient wrt `a` (the only parent of `b`) and store it in your `grads` dictionary, as `grads[a]`. In order to do this, you need the backward function for `func1`, which is stored in the node `b` (recall that the recipe of a tensor can be thought of as a set of instructions for how that tensor was created).""")

    with st.expander("Help - I get AttributeError: 'numpy.ndarray' object has no attribute 'array'"):
        st.markdown("""This might be because you've set `node.grad` to be an array, rather than a tensor. You should store gradients as tensors (think of PyTorch, where `tensor.grad` will have type `torch.Tensor`).
        
It's fine to store numpy arrays in the `grads` dictionary, but when it comes time to set a tensor's grad attribute, you should use a tensor.""")

    with st.expander("Help - I don't know which arguments to pass to my backward functions."):
        st.markdown("""Suppose you are at node `n` during your backprop function's iteration, and you want to use your backward functions to calculate the gradient for node `p`, which is a parent of `n`. The forward function is stored in `n.recipe`, and you can use that to find the appropriate backward function using `BACK_FUNCS`. The arguments you should pass to the backward function are:
        
* `grad_out`, which comes from `grads[n]`
* `x`, which comes from `p`
* `args` and `kwargs`, which are also stored in `n.recipe`""")

    # NEED TO DO THE FOURTH ONE OF MEGS BULLET POINTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def section_more_fwd_bwd():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#non-differentiable-functions">Non-Differentiable Functions</a></li>
    <li><a class="contents-el" href="#negative"><code>negative</code></a></li>
    <li><a class="contents-el" href="#exp"><code>exp</code></a></li>
    <li><a class="contents-el" href="#reshape"><code>reshape</code></a></li>
    <li><a class="contents-el" href="#permute"><code>permute</code></a></li>
    <li><a class="contents-el" href="#expand"><code>expand</code></a></li>
    <li><a class="contents-el" href="#sum"><code>sum</code></a></li>
    <li><a class="contents-el" href="#indexing">Indexing</a></li>
    <li><a class="contents-el" href="#elementwise-add-divide-subtract">Elementwise <code>add</code>, <code>divide</code>, <code>subtract</code></a></li>
    <li><a class="contents-el" href="#in-place-operations">In-Place Operations</a></li>
    <li><a class="contents-el" href="#mixed-scalar-tensor-operations">Mixed scalar-tensor operations</a></li>
    <li><a class="contents-el" href="#in-place-operations">In-Place Operations</a></li>
    <li><a class="contents-el" href="#max"><code>max</code></a></li>
    <li><a class="contents-el" href="#functional-relu">Functional <code>ReLU</code></a></li>
    <li><a class="contents-el" href="#2d-matmul">2D <code>matmul</code></a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""# Filling out the Tensor class with forward and backward methods

Congrats on implementing backprop! The next thing we'll do is write implement a bunch of backward functions that we need to train our model at the end of the day, as well as ones that cover interesting cases.

These should be just like your `log_back` and `multiply_back0`, `multiplyback1` examples earlier.

## Non-Differentiable Functions

For functions like `torch.argmax` or `torch.eq`, there's no sensible way to define gradients with respect to the input tensor. For these, we will still use `wrap_forward_fn` because we still need to unbox the arguments and box the result, but by passing `is_differentiable=False` we can avoid doing any unnecessary computation.

```python
def _argmax(x: Arr, dim=None, keepdim=False):
    '''Like torch.argmax.'''
    return np.expand_dims(np.argmax(x, axis=dim), axis=([] if dim is None else dim))

argmax = wrap_forward_fn(_argmax, is_differentiable=False)
eq = wrap_forward_fn(np.equal, is_differentiable=False)

a = Tensor([1.0, 0.0, 3.0, 4.0], requires_grad=True)
b = a.argmax()
assert not b.requires_grad
assert b.recipe is None
assert b.item() == 3
```

## `negative`

`torch.negative` just performs `-x` elementwise. Make your own version `negative` using `wrap_forward_fn`.

```python
def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backward function for f(x) = -x elementwise.'''
    pass

negative = wrap_forward_fn(np.negative)
BACK_FUNCS.add_back_func(np.negative, 0, negative_back)

utils.test_negative_back(Tensor)
```

## `exp`

Make your own version of `torch.exp`. The backward function should express the result in terms of the `out` parameter - this more efficient than expressing it in terms of `x`.

```python
def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    pass

exp = wrap_forward_fn(np.exp)
BACK_FUNCS.add_back_func(np.exp, 0, exp_back)

utils.test_exp_back(Tensor)
```

## `reshape`

`reshape` is a bit more complicated than the many functions we've dealt with so far: there is an additional positional argument `new_shape`. Since it's not a `Tensor`, we don't need to think about differentiating with respect to it. Remember, `new_shape` is the argument that gets passed into the **forward function**, and we're trying to reverse this operation and return to the shape of the input.

Depending how you wrote `wrap_forward_fn` and `backprop`, you might need to go back and adjust them to handle this. Or, you might just have to implement `reshape_back` and everything will work. 

Note that the output is a different shape than the input, but this doesn't introduce any additional complications.

```python
def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    pass

reshape = wrap_forward_fn(np.reshape)
BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)

utils.test_reshape_back(Tensor)
```

## `permute`

In NumPy, the equivalent of `torch.permute` is called `np.transpose`, so we will wrap that.""")

    with st.expander("""Help - I'm confused about how to implement `permute_back`."""):
        st.markdown("""You should first define the function `invert_transposition`. A docstring is given below:

```python
def invert_transposition(axes: tuple) -> tuple:
    '''
    axes: tuple indicating a transition
    
    Returns: inverse of this transposition, i.e. the array `axes_inv` s.t. we have:
        np.transpose(np.transpose(x, axes), axes_inv) == x
    
    Some examples:
        (1, 0)    --> (1, 0)     # this is reversing a simple 2-element transposition
        (0, 2, 1) --> (0, 1, 2)
        (1, 2, 0) --> (2, 0, 1)  # this is reversing the order of a 3-cycle
    '''
    pass
```

Once you've done this, you can define `permute_back` by transposing again, this time with the inversed transposition instructions.""")

    st.markdown("""
```python
def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    pass

BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
permute = wrap_forward_fn(np.transpose)

utils.test_permute_back(Tensor)
```

## `expand`

Implement your version of `torch.expand`. 

The backward function should just call `unbroadcast`. 

For the forward function, we will use `np.broadcast_to`. This function takes in an array and a target shape, and returns a version of the array broadcasted to the target shape using the rules of broadcasting we discussed in an earlier section. For example:

```python
>>> x = np.array([1, 2, 3])
>>> np.broadcast_to(x, (3, 3))
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])

>>> x = np.array([[1], [2], [3]])
>>> np.broadcast_to(x, (3, 3)) # x is already a column; broadcasting is done along rows
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])
```

The reason we can't just use `np.broadcast_to` and call it a day is that `torch.expand` supports -1 for a dimension size meaning "don't change the size". For example:

```python
>>> x = torch.tensor([[1], [2], [3]])
>>> x.expand(-1, 3)
tensor([[ 1,  1,  1],
        [ 2,  2,  2],
        [ 3,  3,  3]])
```

So when implementing `_expand`, you'll need to be a bit careful when constructing the shape to broadcast to.
""")

    with st.expander("""Help - I'm not sure how to construct the shape."""):
        st.markdown("""If `new_shape` contains no -1s, then you're done. If it does contain -1s, you want to replace those with the appropriate values from `x.shape`.

For example, if `a.shape = (5,)`, and `new_shape = (3, 2, -1)`, you want the actual shape passed into `np.broadcast_to` to be `(3, 2, 5)`. """)

    st.markdown("""
```python
def expand_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    pass

def _expand(x: Arr, new_shape) -> Arr:
    '''Like torch.expand, calling np.broadcast_to internally.

    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    '''
    pass

expand = wrap_forward_fn(_expand)
BACK_FUNCS.add_back_func(_expand, 0, expand_back)

utils.test_expand(Tensor)
utils.test_expand_negative_length(Tensor)
```

## `sum`

The output can also be smaller than the input, such as when calling `torch.sum`. Implement your own `torch.sum` and `sum_back`.

Note, if you get weird exceptions that you can't explain, and these exceptions don't even go away when you use the solutions provided, this probably means that your implementation of `wrap_forward_fn` was wrong in a way which wasn't picked up by the tests. You should return to this function and try to fix it (or just use the solution).

```python
def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False):
    '''Basic idea: repeat grad_out over the dims along which x was summed'''
    pass

def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    '''Like torch.sum, calling np.sum internally.'''
    pass

sum = wrap_forward_fn(_sum)
BACK_FUNCS.add_back_func(_sum, 0, sum_back)

utils.test_sum_keepdim_false(Tensor)
utils.test_sum_keepdim_true(Tensor)
utils.test_sum_dim_none(Tensor)
```

## Indexing

In its full generality, indexing a `torch.Tensor` is really complicated and there are quite a few cases to handle separately.

We only need two cases today:
- The index is an integer or tuple of integers.
- The index is a tuple of (array or Tensor) representing coordinates. Each array is 1D and of equal length. Some coordinates may be repeated. This is [Integer array indexing](https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing).
    - For example, to select the five elements at (0, 0), (1,0), (0, 1), (1, 2), and (0, 0), the index would be the tuple `(np.array([0, 1, 0, 1, 0]), np.array([0, 0, 1, 2, 0]))`. 

Note, in `_getitem` you'll need to deal with one special case: when `index` is of type signature `tuple[Tensor]`. If not for this case, `return x[index]` would suffice for this function. You should define a `coerce_index` function to deal with this particular case; we've provided a docstring for this purpose.
""")

    with st.expander("""Help - I'm confused about the implementation of getitem_back!"""):
        st.markdown("""If no coordinates were repeated, we could just assign the grad for each input element to be the grad at the corresponding output position, or 0 if that input element didn't appear.

Because of the potential for repeat coordinates, we need to sum the grad from each corresponding output position.

Initialize an array of zeros of the same shape as x, and then write in the appropriate elements using `np.add.at`.""")

    st.markdown("""
```python
Index = Union[int, tuple[int, ...], tuple[Arr], tuple[Tensor]]

def coerce_index(index):
    pass

def _getitem(x: Arr, index: Index) -> Arr:
    '''Like x[index] when x is a torch.Tensor.'''
    return x[coerce_index(index)]

def getitem_back(grad_out: Arr, out: Arr, x: Arr, index: Index):
    '''Backwards function for _getitem.

    Hint: use np.add.at(a, indices, b)
    This function works just like a[indices] += b, except that it allows for repeated indices.
    '''
    pass

getitem = wrap_forward_fn(_getitem)
BACK_FUNCS.add_back_func(_getitem, 0, getitem_back)

utils.test_coerce_index(coerce_index, Tensor)
utils.test_getitem_int(Tensor)
utils.test_getitem_tuple(Tensor)
utils.test_getitem_integer_array(Tensor)
utils.test_getitem_integer_tensor(Tensor)
```

## elementwise add, subtract, divide

These are exactly analogous to the multiply case. Note that Python and NumPy have the notion of "floor division", which is a truncating integer division as in `7 // 3 = 2`. You can ignore floor division: - we only need the usual floating point division which is called "true division". 

Use lambda functions to define and register the backward functions each in one line. If you're confused, you can click on the expander below to reveal the first one.""")

    with st.expander("Reveal the first one:"):
        st.markdown("""  
```python
BACK_FUNCS.add_back_func(np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))
```
""")

    st.markdown("""
```python
add = wrap_forward_fn(np.add)
subtract = wrap_forward_fn(np.subtract)
true_divide = wrap_forward_fn(np.true_divide)

# Your code goes here

utils.test_add_broadcasted(Tensor)
utils.test_subtract_broadcasted(Tensor)
utils.test_truedivide_broadcasted(Tensor)
```

## In-Place Operations

Supporting in-place operations introduces substantial complexity and generally doesn't help performance that much. The problem is that if any of the inputs used in the backward function have been modified in-place since the forward pass, then the backward function will incorrectly calculate using the modified version.

PyTorch will warn you when this causes a problem with the error "RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.".

You can implement the warning in the bonus section but for now your system will silently compute the wrong gradients - user beware!

(note - you don't have to fill anything in here; just run the cell)

```python
def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
    '''Like torch.add_. Compute x += other * alpha in-place and return tensor.'''
    np.add(x.array, other.array * alpha, out=x.array)
    return x


def safe_example():
    '''This example should work properly.'''
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    a.add_(b)
    c = a * b
    c.sum().backward()
    assert a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0])
    assert b.grad is not None and np.allclose(b.grad.array, [2.0, 4.0, 6.0, 8.0])


def unsafe_example():
    '''This example is expected to compute the wrong gradients.'''
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


if MAIN:
    safe_example()
    unsafe_example()
```

## Mixed scalar-tensor operations

You may have been wondering why our `Tensor` class has to define both `__mul__` and `__rmul__` magic methods.

Without `__rmul__` defined, executing `2 * a` when `a` is a `Tensor` would try to call `2.__mul__(a)`, and the built-in class `int` would be confused about how to handle this. 

Since we have defined `__rmul__` for you at the start, and you implemented multiply to work with floats as arguments, the following should "just work".

```python
a = Tensor([0, 1, 2, 3], requires_grad=True)
(a * 2).sum().backward()
b = Tensor([0, 1, 2, 3], requires_grad=True)
(2 * b).sum().backward()
assert a.grad is not None
assert b.grad is not None
assert np.allclose(a.grad.array, b.grad.array)
```

## `max`

Since this is an elementwise function, we can think about the scalar case. For scalar $x$, $y$, the derivative for $\max(x, y)$ wrt $x$ is 1 when $x > y$ and 0 when $x < y$. What should happen when $x = y$?

Intuitively, since $\max(x, x)$ is equivalent to the identity function which has a derivative of 1 wrt $x$, it makes sense for the sum of our partial derivatives wrt $x$ and $y$ to also therefore total 1. The convention used by PyTorch is to split the derivative evenly between the two arguments. We will follow this behavior for compatibility, but it's just as legitimate to say it's 1 wrt $x$ and 0 wrt $y$, or some other arbitrary combination that sums to one.
""")

    with st.expander("""Help - I'm not sure how to implement this function."""):
        st.markdown("""Try returning `grad_out * bool_sum`, where `bool_sum` is an array constructed from the sum of two boolean arrays.

You can alternatively use `np.where`.""")

    with st.expander("""Help - I'm passing the first test but not the second."""):
        st.markdown("""This probably means that you haven't implemented `unbroadcast`. You'll need to do this, to get `grad_out` into the right shape before you use it in `np.where`.""")

    st.markdown("""
```python
def maximum_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    '''Backwards function for max(x, y) wrt x.'''
    pass

def maximum_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    '''Backwards function for max(x, y) wrt y.'''
    pass

maximum = wrap_forward_fn(np.maximum)

BACK_FUNCS.add_back_func(np.maximum, 0, maximum_back0)
BACK_FUNCS.add_back_func(np.maximum, 1, maximum_back1)

utils.test_maximum(Tensor)
utils.test_maximum_broadcasted(Tensor)
```

## Functional `ReLU`

A simple and correct ReLU function can be defined in terms of your maximum function. Note the PyTorch version also supports in-place operation, which we are punting on for now.

Again, at $x = 0$ your derivative could reasonably be anything between 0 and 1 inclusive, but we've followed PyTorch in making it 0.5.

```python
def relu(x: Tensor) -> Tensor:
    '''Like torch.nn.function.relu(x, inplace=False).'''
    pass

utils.test_relu(Tensor)
```""")

    st.markdown(r"""
## 2D `matmul`

Implement your version of `torch.matmul`, restricting it to the simpler case where both inputs are 2D.
""")

    with st.expander("Help - I'm confused about matmul2d_back!"):
        st.markdown("""
Try working it out on paper, starting from a couple 2x2 matrices. Can you express the answer in terms of another matrix multiply and a transpose?""")

    with st.expander("Help - I'm still confused about matmul2d_back (and the hint above didn't help)!"):
        st.markdown(r"""
Let $M = X \times Y$, and let $\text{ grad\_out }$ be the gradient of our root node $L$ wrt $M$, i.e.:

$$
\frac{\partial L}{\partial M_{p q}} = \left[\text{ grad\_out }\right]_{p q}\\
$$

Then we have:

$$
\begin{aligned}
\frac{\partial L}{\partial X_{i j}} &=\frac{\partial L}{\partial M_{p q}} \frac{\partial M_{p q}}{\partial X_{i j}} \\
&=\left[\text{ grad\_out }\right]_{p q} \frac{\partial (X_{p r} Y_{r q})}{\partial X_{i j}} \\
&=\left[\text{ grad\_out }\right]_{iq} Y_{j q} \\
&=\left[\text{ grad\_out } \times Y^{\top}\right]_{p q}
\end{aligned}
$$

In other words, the gradient with respect to `x` is `grad\_out @ transpose(y)`.

You can calculate the gradient wrt `y` in the same way.
""")

    st.markdown("""
```python
def _matmul2d(x: Arr, y: Arr) -> Arr:
    '''Matrix multiply restricted to the case where both inputs are exactly 2D.'''
    pass

def matmul2d_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    pass

def matmul2d_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    pass

matmul = wrap_forward_fn(_matmul2d)

BACK_FUNCS.add_back_func(_matmul2d, 0, matmul2d_back0)
BACK_FUNCS.add_back_func(_matmul2d, 1, matmul2d_back1)

utils.test_matmul2d(Tensor)
```
""")

def section_putting_together():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#build-your-own-nn-parameter">Build Your Own <code>nn.Parameter</code></a></li>
    <li><a class="contents-el" href="#build-your-own-linear-layer">Build Your Own Linear Layer</a></li>
    <li><a class="contents-el" href="#build-your-own-cross-entropy-loss">Build Your Own Cross-Entropy Loss</a></li>
    <li><a class="contents-el" href="#build-your-own-no-grad">Build Your Own <code>no_grad</code></a></li>
    <li><a class="contents-el" href="#training-your-network">Training Your Network</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#training-loop">Training Loop</a></li>
    </li></ul>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Putting everything together

## Build Your Own `nn.Parameter`
We've now written enough backwards passes that we can go up a layer and write our own `nn.Parameter` and `nn.Module`.
We don't need much for `Parameter`. It is itself a `Tensor`, shares storage with the provided `Tensor` and requires_grad is `True` by default - that's it!

```python
class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True):
        '''Share the array with the provided tensor.'''
        pass

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"

x = Tensor([1.0, 2.0, 3.0])
p = Parameter(x)
assert p.requires_grad
assert p.array is x.array
assert repr(p) == "Parameter containing:\nTensor(array([1., 2., 3.]), requires_grad=True)"
x.add_(Tensor(np.array(2.0)))
assert np.allclose(
    p.array, np.array([3.0, 4.0, 5.0])
), "in-place modifications to the original tensor should affect the parameter"
```

## Build Your Own `nn.Module`

`nn.Module` is like `torch.Tensor` in that it has a lot of functionality, most of which we don't care about today. We will just implement enough to get our network training. 

Implement the indicated methods (i.e. the ones which are currently just `pass`).

Tip: you can bypass `__getattr__` by accessing `self.__dict__` inside a method.

```python
class Module:
    _modules: dict[str, "Module"]
    _parameters: dict[str, Parameter]

    def __init__(self):
        pass

    def modules(self):
        '''Return the direct child modules of this module.'''
        return self.__dict__["_modules"].values()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        '''Return an iterator over Module parameters.

        recurse: if True, the iterator includes parameters of submodules, recursively.
        '''
        pass

    def __setattr__(self, key: str, val: Any) -> None:
        '''
        If val is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call the superclass.
        '''
        pass

    def __getattr__(self, key: str) -> Union[Parameter, "Module"]:
        '''
        If key is in _parameters or _modules, return the corresponding value.
        Otherwise, raise KeyError.
        '''
        pass

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


class TestInnerModule(Module):
    def __init__(self):
        super().__init__()
        self.param1 = Parameter(Tensor([1.0]))
        self.param2 = Parameter(Tensor([2.0]))

class TestModule(Module):
    def __init__(self):
        super().__init__()
        self.inner = TestInnerModule()
        self.param3 = Parameter(Tensor([3.0]))

mod = TestModule()
assert list(mod.modules()) == [mod.inner]
assert list(mod.parameters()) == [
    mod.param3,
    mod.inner.param1,
    mod.inner.param2,
], "parameters should come before submodule parameters"
print("Manually verify that the repr looks reasonable:")
print(mod)
```

## Build Your Own Linear Layer

You may have a `Linear` written already that you can adapt to use our own `Parameter`, `Module`, and `Tensor`. If your `Linear` used `einsum`, use a `matmul` instead. You can implement a backward function for `einsum` in the bonus section.

```python
class Linear(Module):
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        pass

    def forward(self, x: Tensor) -> Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        pass

    def extra_repr(self) -> str:
        pass
```

""")

    with st.expander("""Help - I get "AttributeError: 'numpy.ndarray' object has no attribute 'array'"."""):
        st.markdown("""This is probably because you've multiplied a tensor by a numpy array during your initialisation of weights or biases. This currently can't be handled by your code. You should multiply by a float instead.""")

    st.markdown("""Now we can define a MLP suitable for classifying MNIST, with zero PyTorch dependency!

```python
class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(28 * 28, 64)
        self.linear2 = Linear(64, 64)
        self.output = Linear(64, 10)

    def forward(self, x):
        x = x.reshape((x.shape[0], 28 * 28))
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = self.output(x)
        return x
```

## Build Your Own Cross-Entropy Loss

Make use of your integer array indexing to implement `cross_entropy`. See the documentation page [here](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

```python
def cross_entropy(logits: Tensor, true_labels: Tensor) -> Tensor:
    '''Like torch.nn.functional.cross_entropy with reduction='none'.

    logits: shape (batch, classes)
    true_labels: shape (batch,). Each element is the index of the correct label in the logits.

    Return: shape (batch, ) containing the per-example loss.
    '''
    pass

utils.test_cross_entropy(Tensor, cross_entropy)
```

## Build Your Own `no_grad`

The last thing our backpropagation system needs is the ability to turn it off completely like `torch.no_grad`. 

Below, we have an implementation of the `NoGrad` context manager so that it reads and writes the `grad_tracking_enabled` flag from the top of the file. In general, using mutable global variables is not ideal because multiple threads will be a problem, but we will leave that for another day.

```python
class NoGrad:
    '''Context manager that disables grad inside the block. Like torch.no_grad.'''

    was_enabled: bool

    def __enter__(self):
        "SOLUTION"
        global grad_tracking_enabled
        self.was_enabled = grad_tracking_enabled
        grad_tracking_enabled = False

    def __exit__(self, type, value, traceback):
        "SOLUTION"
        global grad_tracking_enabled
        grad_tracking_enabled = self.was_enabled
```

## Training Your Network

We've already looked at data loading and training loops earlier in the course, so we'll provide a minimal version of these today as well as the data loading code.

```python
(train_loader, test_loader) = utils.get_mnist(20)
utils.visualize(train_loader)

class SGD:
    def __init__(self, params: Iterable[Parameter], lr: float):
        '''Vanilla SGD with no additional features.'''
        self.params = list(params)
        self.lr = lr
        self.b = [None for _ in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        with NoGrad():
            for (i, p) in enumerate(self.params):
                assert isinstance(p.grad, Tensor)
                p.add_(p.grad, -self.lr)


def train(model, train_loader, optimizer, epoch):
    for (batch_idx, (data, target)) in enumerate(train_loader):
        data = Tensor(data.numpy())
        target = Tensor(target.numpy())
        optimizer.zero_grad()
        output = model(data)
        loss = cross_entropy(output, target).sum() / len(output)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, test_loader):
    test_loss = 0
    correct = 0
    with NoGrad():
        for (data, target) in test_loader:
            data = Tensor(data.numpy())
            target = Tensor(target.numpy())
            output = model(data)
            test_loss += cross_entropy(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == target.reshape(pred.shape)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
```

### Training Loop

To finish the day, let's see if everything works correctly and our MLP learns to classify MNIST. It's normal to encounter some bugs and glitches at this point - just go back and fix them until everything runs.

```python
num_epochs = 5
model = MLP()
start = time.time()
optimizer = SGD(model.parameters(), 0.01)
for epoch in range(num_epochs):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)
    optimizer.step()
print(f"Completed in {time.time() - start: .2f}s")
```
""")

def section_bonus():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#in-place-operation-warnings">In-Place Operation Warnings</a></li>
    <li><a class="contents-el" href="#in-place-relu">In-Place <code>ReLU</code></a></li>
    <li><a class="contents-el" href="#backward-for-einsum">Backward for <code>einsum</code></a></li>
    <li><a class="contents-el" href="#convolutional-layers">Convolutional layers</a></li>
    <li><a class="contents-el" href="#resnet-support">ResNet Support</a></li>
    <li><a class="contents-el" href="#central-difference-checking">Central Difference Checking</a></li>
    <li><a class="contents-el" href="#non-differentiable-function-support">Non-Differentiable Function Support</a></li>
    <li><a class="contents-el" href="#differentiation-wrt-keyword-arguments">Differentiation wrt Keyword Arguments</a></li>
    <li><a class="contents-el" href="#torch-stack"><code>torch.stack</code></a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""

## Bonus
Congratulations on finishing the day's main content! 

### In-Place Operation Warnings

The most severe issue with our current system is that it can silently compute the wrong gradients when in-place operations are used. Have a look at how [PyTorch handles it](https://pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd) and implement a similar system yourself so that it either computes the right gradients, or raises a warning.

### In-Place `ReLU`
Instead of implementing ReLU in terms of maximum, implement your own forward and backward functions that support `inplace=True`.

### Backward for `einsum`
Write the backward pass for your equivalent of `torch.einsum`.

### Reuse of Module during forward
Consider the following MLP, where the same `nn.ReLU` instance is used twice in the forward pass. Without running the code, explain whether this works correctly or not with reference to the specifics of your implementation.

```python
class MyModule(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(28*28, 64)
        self.linear2 = Linear(64, 64)
        self.linear3 = Linear(64, 10)
        self.relu = ReLU()
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)
```

### Convolutional layers
Now that you've implemented a linear layer, it should be relatively straightforward to take your convolutions code from day 2 and use it to make a convolutional layer. How much better performance do you get on the MNIST task once you replace your first two linear layers with convolutions?

### ResNet Support
Make a list of the features that would need to be implemented to support ResNet inference, and training. It will probably take too long to do all of them, but pick some interesting features to start implementing.

### Central Difference Checking
Write a function that compares the gradients from your backprop to a central difference method. See [Wikipedia](https://en.wikipedia.org/wiki/Finite_difference) for more details.

### Non-Differentiable Function Support
Your `Tensor` does not currently support equivalents of `torch.all`, `torch.any`, `torch.floor`, `torch.less`, etc. which are non-differentiable functions of Tensors. Implement them so that they are usable in computational graphs, but gradients shouldn't flow through them (their contribution is zero).

### Differentiation wrt Keyword Arguments
In the real PyTorch, you can sometimes pass tensors as keyword arguments and differentiation will work, as in `t.add(other=t.tensor([3,4]), input=t.tensor([1,2]))`. In other similar looking cases like `t.dot`, it raises an error that the argument must be passed positionally. Decide on a desired behavior in your system and implement and test it.

### `torch.stack`
So far we've registered a separate backwards for each input argument that could be a Tensor. This is problematic if the function can take any number of tensors like `torch.stack` or `numpy.stack`. Think of and implement the backward function for stack. It may require modification to your other code.
""")

func_list = [section_home, section_intro, section_autograd, section_more_fwd_bwd, section_putting_together, section_bonus]

page_list = ["🏠 Home", "1️⃣ Introduction", "2️⃣ Autograd", "3️⃣ More forward & backward functions", "4️⃣ Putting everything together", "5️⃣ Bonus"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()
