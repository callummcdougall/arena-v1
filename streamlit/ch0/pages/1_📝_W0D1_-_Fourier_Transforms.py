import streamlit as st

st.set_page_config(layout="wide")

import platform
is_local = (platform.processor() != "")
rootdir = "" if is_local else "ch0/"

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
## 1️⃣ Fourier Transforms

The first set of exercises covers Fourier Transforms, using just the NumPy library. This should get you comfortable with the basic idea of working through exercises, and will also introduce some of the concepts that recur in part 2 of the exercises, where you will crete your own simple neural networks to fit functions to arbitrary polynomials.

This part should take you **2-3 hours**.

## 2️⃣ Basic Neural Network

Here, we'll start to write up an actual neural network which builds on the work we've done in part 1.

This part should take you **1-2 hours**.

## 3️⃣ Bonus Exercises

If you get through the first two sections, you can use some of the suggestions here as jumping off points to explore some of the exercises in greater detail.
""")

def section_fourier():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#recommended-reading">Recommended reading</a></li>
    <li><a class="contents-el" href="#imports">Imports</a></li>
    <li><a class="contents-el" href="#discrete-fourier-transform">Discrete Fourier Transform</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#exercise-1-dft">Exercise 1 - DFT</a></li>
        <li><a class="contents-el" href="#aside-typing">Aside - typing</a></li>
        <li><a class="contents-el" href="#exercise-2-inverse-dft">Exercise 2 - inverse DFT</a></li>
        <li><a class="contents-el" href="#aside-testing">Aside - testing</a></li>
        <li><a class="contents-el" href="#exercise-3-test-functions">Exercise 3 - test functions</a></li>
    </ul></li>
    <li><a class="contents-el" href="#continuous-fourier-transform">Continuous Fourier Transform</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#exercise-1-intergration">Exercise 1 - Integration</a></li>
        <li><a class="contents-el" href="#exercise-2-fourier-series">Exercise 2 - Fourier series</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""# Fourier Transforms

Fourier transforms are an interesting branch of mathematics which will crop up in several places later in this programme. For instance, they are used in feature visualisation because they often provide a more natural basis than the standard one. Additionally, the discrete Fourier transform was recently featured in Neel Nanda's [Grokking study](https://www.lesswrong.com/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking), which we will look at in the interpretability week.

## Recommended reading

* [3Blue1Brown video](https://www.youtube.com/watch?v=spUNpyF58BY&vl=en) on Fourier transforms
* [An Interactive Guide To The Fourier Transform](https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/)

## Imports

```python
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum

import utils
```

## Discrete Fourier Transform""")

    st.markdown(r"""

Given a tuple of complex numbers $(x_0, x_1, ..., x_{N-1})$, it's **discrete Fourier transform** (DFT) is the sequence $(y_0, y_1, ..., y_{N-1})$ defined by: 
$$
y_k=\sum_{j=0}^{N-1} \omega_N^{jk} x_j
$$
where $\omega_N = e^{-2\pi i/N}$ is the **Nth root of unity**.

This can equivalently be written as the following matrix equation:""")

    st.latex(r"""\left[\begin{array}{ccccc}
1 & 1 & 1 & \ldots & 1 \\
1 & \omega_N & \omega_N^{2} & \ldots & \omega_N^{N-1} \\
1 & \omega_N^{2} & \omega_N^{4} & \ldots & \omega_N^{2(N-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega_N^{N-1} & \omega_N^{2(N-1)} & \ldots & \omega_N^{(N-1)^2}
\end{array}\right]\left[\begin{array}{c}
x_0 \\
x_1 \\
\vdots \\
x_{N-1}
\end{array}\right]=\left[\begin{array}{c}
y_0 \\
y_1 \\
\vdots \\
y_{N-1}
\end{array}\right]
""")

    st.markdown("""### Exercise 1 - DFT

Write a function which calculates the DFT of an array `x`, using the matrix equation above.""")

    st.code('''
def DFT_1d(arr : np.ndarray) -> np.ndarray:
    """
    Returns the DFT of the array `arr`, using the equation above.
    """
    pass''')

    st.markdown("""###### """)

    with st.expander("Help - I don't know how to implement complex numbers."):
        st.markdown("Python represents complex numbers using the letter `j`. For instance, we can use `2j * np.pi` to represent the complex number $2\\pi i$")

    with st.expander("Help - I'm not sure how to construct the left matrix."):
        st.markdown("Try first making the exponents, using `np.outer`. Then take advantage of NumPy's vectorisation to make the full matrix.")

    st.markdown("""

### Aside - typing

The `typing` module is extremely useful when catching errors in your code, when used alongside VSCode's type checker extension.

You can activate typing by going to the `settings.json` file in VSCode, and adding this line:""")

    st.code('''{
    "python.analysis.typeCheckingMode": "basic"
}''')

    st.markdown("""You can open the `settings.json` file by first opening VSCode's Command Palette (shortcut `Shift + Cmd + P` for Mac, `Ctrl + Shift + P` for Windows/Linux), then finding the option **Preferences: Open User Settings (JSON)**.

If you're finding that the type checker is throwing up too many warnings, you can suppress them by adding the comment `# type: ignore` at the end of a line, or using the `cast` function from the `typing` library to let the Python interpreter know what type it's dealing with. However, in general you should try and avoid doing this, because type checker warnings usually mean there's a better way for you to be writing your code. An (often better) solution is to add in an `assert isinstance` statement - for instance, if the output of a function should be an integer but the type checker is complaning because it doesn't "know" the output is an integer, the line `assert isinstance(out, int)` should remove the red line.""")

    st.markdown("""### Exercise 2 - Inverse DFT

Now try to write the same function, but with an optional `inverse` argument. If this is true, you should return the inverse discrete Fourier transform (the equation for which can be found [here](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Inverse_transform)).

The code below also includes a test function. If you run this and it raises an error, then it probably means there's a mistake somewhere in your code. If there is no output, then the code is working as expected.
""")

    st.code('''def DFT_1d(arr: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Returns the DFT of the array `arr`, with an optional `inverse` argument.
    """
    pass
        
utils.test_DFT_func(DFT_1d)''')

    st.markdown("""

### Aside - testing

This is the first of many test functions you'll see during this programme. If you right-click on the function as it appears in your code, and select **Go to Definition**, you will see this:""")

    st.code("""def test_DFT_func(DFT_1d, x=np.linspace(-1, 1), function=np.square) -> None:
    
    y = function(x)
    
    y_DFT_actual = DFT_1d(y)
    y_reconstructed_actual = DFT_1d(y_DFT_actual, inverse=True)
    
    y_DFT_expected = np.fft.fft(y)
    
    np.testing.assert_allclose(y_DFT_actual, y_DFT_expected, atol=1e-10, err_msg="DFT failed")
    np.testing.assert_allclose(y_reconstructed_actual, y, atol=1e-10, err_msg="Inverse DFT failed")

""")

    st.markdown("""Let's briefly go through this function to explain what it does.

The first few lines define `y` as some function of the array `x` (by default `y = x^2`), then using your function DFT_1d` to apply the DFT, then inverse DFT. It then calculates the DFT using NumPy's function (which is assumed to be accurate). Finally, the last two lines raise an error if the values produced by your function are different from the values produced by NumPy's functions.

Not all tests will be perfect, and there might be errors that they miss. For instance, consider the test below. What is the problem with using this to check that your implementation is correct?

```python
def test_DFT_func_bad(DFT_1d, x=np.linspace(-1, 1), function=np.square) -> None:
    
    y = function(x)
    y_DFT = DFT_1d(y)
    y_reconstructed = DFT_1d(y_DFT, inverse=True)
    
    np.testing.assert_allclose(y, y_reconstructed, atol=1e-10)
```
""")

    with st.expander("Reveal answer"):
        st.markdown("""This test only checks whether the `inverse=True` argument causes `DFT_1d` to perform the inverse operation, not whether the original operation was actually the intended one. For instance, if your `DFT_1d` actually performed an identity mapping, this would pass the test.""")

    st.markdown("""
As the coding we do becomes more complicated, these kinds of issues will become more common. It will often be necessary to perform your own tests to verify that your outputs are correct, which apply more stringent tests than the functions we provide. It can be very frustrating to pass all of the tests, only for your code to fail later because the tests weren't good enough to catch all possible errors!

### Exercise 3 - test functions

Write your own test function for `DFT_1d`. 

Rather than using NumPy's built-in DFT function, this test should verify the behaviour of `DFT_1d` on a particular input, where the behaviour is known. For instance, the [Wikipedia page](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Example) gives an example signal, and its DFT. You can use `np.testing.assert_allclose` to check that you get the expected output.
""")

    st.markdown("""## Continuous Fourier Transform

In subsequent exercises, we'll work with continuous Fourier transforms rather than discrete. 

The DFT worked with a finite set of sampled values of a particular function, and allowed us to produce frequencies up to a finite maximum. In contrast, the continuous Fourier transform takes a real-valued function and can return the amplitude of any frequency.

### Exercise 1 - Integration

First, we'll build up a few functions to help us. The next two functions calculate an integral, and the product of two functions respectively (this will be useful when calculating Fourier coefficients).
""")

    st.code('''def integrate_function(func: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Calculates the approximation of the Riemann integral of the function `func`, 
    between the limits x0 and x1.
    
    You should use the Left Rectangular Approximation Method (LRAM).
    """

    pass

utils.test_integrate_function(integrate_function)''')

    st.code('''def integrate_product(func1: Callable, func2: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Computes the integral of the function x -> func1(x) * func2(x).
    """
    
    pass

utils.test_integrate_product(integrate_product)''')

    st.markdown(r"""Now, we will write a function which computes the Fourier coefficients. These are terms $(a_n)_{n\geq 0}, (b_n)_{n\geq 1}$ s.t. we can write any sufficiently well-behaved function as:
$$
f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty}a_n \cos{nx} + \sum_{n=1}^{\infty}b_n \sin{nx}
$$

You can find the formula for these coefficients on [Wolfram Alpha](https://mathworld.wolfram.com/FourierSeries.html).

We can also get an approximation to $f(x)$, by truncating this Fourier series after a finite number of terms (i.e. up to some maximum frequency $N$):
$$
\hat{f}_N(x) = \frac{a_0}{2} + \sum_{n=1}^{N}a_n \cos{nx} + \sum_{n=1}^{N}b_n \sin{nx}
$$

### Exercise 2 - Fourier series
""")


    st.code('''def calculate_fourier_series(func: Callable, max_freq: int = 50):
    """
    Calculates the fourier coefficients of a function, 
    assumed periodic between [-pi, pi].
    
    Your function should return ((a_0, A_n, B_n), func_approx), where:
        a_0 is a float
        A_n, B_n are lists of floats, with n going up to `max_freq`
        func_approx is the fourier approximation, as described above
    """

    pass

step_func = lambda x: 1 * (x > 0)
create_interactive_fourier_graph(calculate_fourier_series, func = step_func)''')

    with st.expander("Help - I'm having trouble calculating the coefficients."):
        st.markdown("""To get `a_n`, try using `integrate_product` with the functions `f` and `lambda x: np.cos(n*x)`.""")


    st.markdown("""---

If this code has been written correctly, then when run it should produce interactive output that looks like this:
    """)

    st.image(rootdir + "images/ani1.png")

    st.markdown(r"""
You should be able to move the slider to see how the Fourier series converges to the true function over time.

You can change the `func` parameter in `create_interactive_fourier_graph`, and investigate some different behaviour. Here are a few you might want to try:

* Polynomials of different order
* Piecewise linear functions, e.g. the sawtooth
* Linear combinations of trig terms, e.g. $\sin{3x} + \cos{17x}$. What do you expect to see in these cases?
""")

    with st.expander("Explanation for sin(3x) + cos(17x)"):
        st.markdown("""You should see something like this:""")
        st.image(rootdir + "images/ani1a.png")
        st.image(rootdir + "images/ani1b.png")
        st.image(rootdir + "images/ani1c.png")
        st.markdown(r"""This is because the Fourier series are orthogonal in the range $[-\pi, \pi]$.""")
        st.markdown("""The only non-zero coefficients are the ones that exactly match the frequencies already present in the data, so we only get changes in the reconstructed function once we add the 3rd and 17th frequencies.""")

    st.markdown("""Use the sidebar to navigate to part 2 of today's exercises.""")

def section_nn():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#i-numpy"><code>(I)</code> NumPy</a></li>
    <li><a class="contents-el" href="#ii-pytorch-tensors"><code>(II)</code> PyTorch & Tensors</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#tensor-basics">Tensor basics</a></li>
        <li><a class="contents-el" href="#how-to-create-tensors">How to create tensors</a></li>
        <li><a class="contents-el" href="#exercise-refactor-your-code-ii">Exercise - refactor your code <code>(II)</code></a></li>
    </ul></li>
    <li><a class="contents-el" href="#iii-autograd"><code>(III)</code> Autograd</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#exercise-refactor-your-code-iii">Exercise - refactor your code <code>(III)</code></a></li>
    </ul></li>
    <li><a class="contents-el" href="#iv-models"><code>(IV)</code> Models</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#exercise-refactor-your-code-iv">Exercise - refactor your code <code>(IV)</code></a></li>
    </ul></li>
    <li><a class="contents-el" href="#iii-optimizers"><code>(V)</code> Optimizers</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#exercise-refactor-your-code-v">Exercise - refactor your code <code>(V)</code></a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""
# Basic Neural Network

Here, we'll start to write up an actual neural network which builds on the work we've done in part 1.

We will start by using only NumPy, working from first principles, and slowly add more elements of PyTorch until we're using a full neural network.

This is the simplest possible neural network architecture - it only has a single layer, and only uses linear functions. All we are doing is learning the coefficients of a Fourier series. The inputs to our network are the frequencies, and the weights are the coefficients, so our output is the same as the truncated Fourier series expression we saw in the previous section:""")

    st.image(rootdir + "images/diagram.png")

    st.markdown("""
How can we learn these weights? Well, it turns out that the Fourier series coefficients are also minimisers of the **Mean Squared Error** (MSE) between the Fourier series and the original function. The mathematical reason for this is that taking a finite Fourier series actually orthogonally projects our function onto a different basis, which also minimises the MSE. So if we calculate the MSE, and differentiate it with respect to our coefficients, this will tell us how to adjust the coefficients in a way which makes the error smaller, thus moving our coefficients closer to the true Fourier series.

#### Question - what is the derivative of the quadratic loss for a single sample, wrt the weights?

Try and derive the answer before you reveal it below.""")

    with st.expander("Reveal answer"):
        st.markdown(r"""Our loss function is:
$$
L=(y-\hat{y})^2
$$
and the derivative of this wrt our prediction $\hat{y}$ is:
$$
\frac{dL}{d\hat{y}} = 2(\hat{y}-y)
$$
Furthermore, we can also easily find the gradient of our prediction $\hat{y}$ relative to the coefficients:
$$
\begin{aligned}
&\frac{d\hat{y}}{d \boldsymbol{a}}=\left[\begin{array}{c}
\frac{1}{2} \\
\cos x \\
\cos 2 x
\end{array}\right], \;\;
\frac{d\hat{y}}{d \boldsymbol{b}}=\left[\begin{array}{c}
\sin x \\
\sin 2 x
\end{array}\right]
\end{aligned}
$$
so if we use the **chain rule**, then we can find the gradient of the loss wrt each of our coefficients:
$$
\begin{aligned}
&\frac{dL}{d \boldsymbol{a}}=2(\hat{y}-y)\left[\begin{array}{c}
\frac{1}{2} \\
\cos x \\
\cos 2 x
\end{array}\right], \;\;
\frac{dL}{d \boldsymbol{b}}=2(\hat{y}-y)\left[\begin{array}{c}
\sin x \\
\sin 2 x
\end{array}\right]
\end{aligned}
$$""")

    st.markdown(r"""
## (I) NumPy

Here, you should fill in a function which performs gradient descent on the coefficients of your function. At each step, we will calculate the total squared error between the true function `y` and your prediction `y_pred` over the range $[-\pi, \pi]$. We then manually implement gradient descent on our learned coefficients, moving them closer to the ideal values.

You have the global variables `TARGET_FUNC` (the function you should be trying to approximate), `NUM_FREQUENCIES` (which corresponds to our value $N$ in the truncated Fourier series expression that we saw in the previous section), and the **hyperparameters** `TOTAL_STEPS` and `LEARNING_RATE` which control how gradient descent is implemented.

Lots of this function is already filled in for you, but some sections are replaced with comments saying `# TODO`, followed by a description of what should go in this section. These are the sections of code you need to fill in (replacing the `raise Exception` statement with your own code).

Note - you might find the library `einsum` useful here. You can read up on how to use it [here](https://rockt.github.io/2018/04/30/einsum). We will also cover it in more depth tomorrow.
    """)

    st.code("""
NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = np.linspace(-np.pi, np.pi, 2000)
y = TARGET_FUNC(x)

x_cos = np.array([np.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = np.array([np.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = np.random.randn()
A_n = np.random.randn(NUM_FREQUENCIES)
B_n = np.random.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    
    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    raise Exception("Not yet implemented.")

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    raise Exception("Not yet implemented.")
    
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
        y_pred_list.append(y_pred)
    
    # TODO: compute gradients of coeffs with respect to `loss`
    raise Exception("Not yet implemented.")

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    raise Exception("Not yet implemented.")

utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
""")

    with st.expander("""Help - I'm not sure how to compute the gradients wrt loss."""):
        st.markdown(r"""You should refer back to the **Question - what is the derivative of the quadratic loss for a single sample, wrt the weights?** section above.
        
You may find it helpful to first define `grad_y_pred` as the derivative of $L$ wrt $\hat{y}$, and then calculating the gradients wrt each of the weights.""")

    st.markdown("""
If this works, then you should see a graph with a slider, that you can move to see the convergence of your function to the target one over time (along with a changing title to represent the coefficients):
""")

    st.image(rootdir + "images/ani2.png")

    st.markdown("""## (II) PyTorch & Tensors

### Tensor basics

Tensors are the standard object in PyTorch, analogous to arrays in NumPy. However, they come with several additional features, most notably:

* They can be moved to the GPU, for much faster computation
* They can store gradients as computations are performed on them, which enables backpropogation in neural networks

Fortunately, many of the ways of working with tensors carry over quite nicely from NumPy arrays. A few differences are:

* There's some additional subtlety in how to create tensors (see next section).
* Many PyTorch functions take an optional keyword argument `out`. If provided, instead of allocating a new tensor and returning that, the output is written directly to the out tensor.
* PyTorch tends to use the keyword argument `dim` where NumPy uses `axis`.
* Not all functions have the same name (e.g. the equivalent of `np.concatenate` is `torch.cat`).

If you haven't already, this would be a good time to review the [100 NumPy exercises](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.ipynb), and work through them using PyTorch. This should get you a lot more fluent in how to create and manipulate tensors. If one member of your pair has done these exercises but the other hasn't, it's fine to just read over the solutions.

### How to create tensors

Two ways to create objects of type `torch.Tensor` are:

* Call the constructor of `torch.Tensor`
* Use the creation function `torch.tensor`

The constructor way is fraught with peril. Try running the following code:""")

    st.code("""import torch
x = torch.arange(5)
y1 = torch.Tensor(x.shape)
y2 = torch.Tensor(tuple(x.shape))
y3 = torch.Tensor(list(x.shape))
print(y1, y2, y3)""")

    st.markdown("""Why is this output weird? The argument to `torch.Tensor` can be interpreted in one of two ways:

* As the tensor's **shape**, in which case it acts like `torch.empty`; returning a tensor of the given shape & filled with uninitialised data
    * This happens for `y1` in the example above
* As the tensor's **input data**, in which case it acts equivalently to NumPy's `np.array` function when you pass it a list or tuple
    * This happens for `y2` and `y3`

Becuase of this ambiguity, it's usually best to use `torch.tensor`, which always takes input data rather than a shape. `t.tensor` with no dtype specified will try to detect the type of your input automatically. This is usually what you want, but not always. For example, what does the following code do?""")

    st.code("""try:
    print(torch.tensor([1, 2, 3, 4]).mean())
except Exception as e:
    print("Exception raised: ", e)
""")

    st.markdown("""NumPy's `np.mean` would coerce to float and return `2.5` here, but PyTorch detects that your inputs all happen to be integers and refuses to compute the mean because it's ambiguous if you wanted `2.5` or `10 // 4 = 2` instead.

The best practice to avoid surprises and ambiguity is to use `torch.tensor` and pass the dtype explicitly.

One final gotcha with `torch.tensor` - in NumPy you can create a 2D array using `np.array(array_list)`, where `array_list` is a list of arrays. In PyTorch, you can't do this with `torch.tensor`. The best way to convert a list of equal-length 1D tensors into a 2D tensor is by using `torch.stack`.

Other good ways to create tensors are:

* If you already have a tensor `input` and want a new one of the same size, use functions like `torch.zeros_like(input)`. This uses the dtype and device of the input by default, saving you from manually specifying it.
* If you already have a tensor `input` and want a new one with the same dtype and device but new data, use the `input.new_tensor` method.
Many [other creation functions](https://pytorch.org/docs/stable/torch.html#creation-ops) exist, for which you should also specify the dtype explicitly.

NumPy arrays can be converted into tensors using `torch.from_numpy`. Tensors can be converted into numpy arrays using the tensor method `.numpy()`.

One final note - since we'll be using `torch` functions frequently, we'll be using the convention here of `import torch as t`.

### Exercise - refactor your code (II)

Rewrite the code from the previous exercise, but use PyTorch tensors rather than NumPy arrays. Your final code shouldn't contain any instances of `np`.

Some tips here:
* Remember you can use `torch.stack` to create 2D arrays
* You should still append numpy arrays rather than tensors to `coeffs_list`. You can use `.numpy()` for this.

You can move on when your code successfully produces the same graphical output as it did before.""")

    with st.expander("Help - the cos and sin coefficients in my title aren't changing."):
        st.markdown("""This is probably because you've appended the original tensor to `y_pred_list`. Try to append a copy instead.""")

    st.markdown("""## (III) Autograd

We'll be covering autograd and the backpropagation mechanism a lot more in subsequent days and weeks. For now, we'll keep things relatively straightforward.

Rather than manually computing gradients, PyTorch keeps track of operations performed on tensors and stores those gradients within the tensors themselves (provided the tensor in question was initialised with `requires_grad=True`). The gradients can be accessed using the `grad` attribute. An example:
""")

    st.code("""import torch

a = torch.tensor(2, dtype=torch.float, requires_grad=True)
b = torch.tensor(3, dtype=torch.float, requires_grad=True)

Q = 3*a**3 - b**2""")

    st.markdown(r"""Note that we require `a` and `b` to have `float` dtypes, since we can't propagate gradients with `int` dtypes.

We have created `Q` as a function of tensors `a` and `b`. When this happens, PyTorch keeps track of the operations performed on `Q`. We can explicitly calculate the gradients ourselves:
$$

\begin{aligned}
Q &= 3a^3 - b^2 \\
\\
\frac{\partial Q}{\partial a} &=9 a^2 \\
\frac{\partial Q}{\partial b} &=-2 b
\end{aligned}

$$""")

    st.markdown("""
When we call `.backward()` on `Q`, autograd calculates these gradients and stores them in the respective tensors’ `.grad` attribute.

Note that, if `Q` had more than one element, we would need to consider directional derivatives (see [this page](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) for more information). But here `Q` only has a single element, so we don't need to worry about this.
""")

    st.code("""
# check if collected gradients are correct
assert 9*a**2 == a.grad
assert -2*b == b.grad
""")

    st.markdown("""
### Exercise - refactor your code (III)

Use autograd to refactor the code you wrote in the previous section, so that gradient descent is implemented using autograd. This will mainly involve rewriting the last two `# TODO`'s in the code.

Additionally, you will also need to be a bit more careful when adding your tensors to `y_pred_list`. Rather than just calling `.numpy()` on our tensor, we first have to call `.detach()`. A description of what `detach` does can be found [here](https://www.tutorialspoint.com/what-does-tensor-detach-do-in-pytorch). We will go into more detail when we study backpropagation later on in this course.

A final note - after you apply the GD step, you'll need to reset the gradients of your parameters. If you don't do this, PyTorch will keep accumulating gradients every time you run `backward()`. In later sections we'll see more advanced ways to reset gradients, but for now you can just call `param.grad = None` to reset the gradients of `param`.

## (IV) Models

Large neural networks are often structured in **layers**. Many of these layers have **learnable parameters** which will be optimised during learning. PyTorch gives us the `nn`. package to arrange the computation into layers, thereby abstracting away much of the difficulty for us.

The `nn` package defines a set of **Modules**, which are roughly equivalent to neural network layers. A Module receives input Tensors and computes output Tensors, but may also hold internal state such as Tensors containing learnable parameters. If you have more than one layer, you can combine them into a single object using `nn.Sequential(*args)`.

The `nn` package also defines a set of useful loss functions that are commonly used when training neural networks.

### Exercise - refactor your code (IV)

Now, you should rewrite your previous code to make use of the `nn` library. In other words, rather than defining parameters `a_0`, `A_n` and `B_n` explicitly, you can define a linear layer `nn.Linear`, and have the parameters correspond to the weights of that network.

This will probably be the most difficult refactoring so far, especially if you don't have much experience working with `torch.nn`. A few notes here:

* Once you define `model`, you can create output by simply calling `model(input)`. Previously you had three inputs: a constant term, `x_cos` and `x_sin`. Since a linear layer comes with a bias term by default, we only have to worry about `x_cos` and `x_sin`. You'll need to combine these into a single array before feeding it into your model.
* When neural networks get multi-dimensional input, they always interpret the `0`th dimension as the **batch dimension**. In this case, the batch dimension is the linspace of the `x` tensor (since for every possible value in this tensor, our network is computing a single output which will be a value in the tensor `y_pred`). So your input to the neural network should have shape `(2000, 2 * NUM_FREQUENCIES)`.
* You can also add an instance of `nn.Flatten` to go after your linear layer, which does something similar to the `squeeze` method for NumPy arrays. Look at the PyTorch documentation page for more.
* `model.parameters()` returns an iterator, which you can loop through to apply gradient descent to each `param`.
    * It's good practice to use `torch.no_grad()` during the gradient descent step, to make sure that you don't propagate gradients during this step. `torch.inference_mode()` does the same thing, and is currently preferred (you can see [this documentation page](https://pytorch.org/docs/stable/generated/torch.autograd.inference_mode.html#:~:text=InferenceMode%20is%20a%20new%20context,tracking%20and%20version%20counter%20bumps.) for more details).
* You will also need to reset gradients before each step of gradient descent. You can do this in the for loop above by setting `param.grad = None`, but PyTorch provides an easier method: calling `model.zero_grad()` resets the gradients of all parameters in the model.
* Streamlit

Again, you'll be finished once you can produce the same output as before.

## (V) Optimizers

Up to this point we have updated the weights of our models by manually mutating the Tensors holding learnable parameters with torch.no_grad(). This is not a huge burden for simple optimization algorithms like stochastic gradient descent, but in practice we often train neural networks using more sophisticated optimizers like AdaGrad, RMSProp, Adam, etc.

The optim package in PyTorch abstracts the idea of an optimization algorithm and provides implementations of commonly used optimization algorithms.

In this example we will use the nn package to define our model as before, but we will optimize the model using the SGD algorithm provided by the optim package.

### Exercise - refactor your code (V)

Refactor your code for a final time. You won't need as many changes as you did last time; you'll just have to:
* Define an optimizer using SGD, with first argument the model parameters and second argument learning rate (see [this documentation page](https://pytorch.org/docs/stable/optim.html)).
* Remove the gradient descent step from your previous code (which should have involved a for loop), and replace it with a single line of code involving an optimizer.

Note - you can also replace `model.zero_grad()` with `optimizer.zero_grad()`. This is functionally the same here, because the optimizer is fed all the parameters of the model. However this might not always be the case (e.g. when finetuning a model, you might only want to optimise the final layer). You might also have more than one optimiser for the same model. In these cases, it is often safer to call `model.zero_grad()`, so that every gradient is reset.

""")

def section_bonus():
    st.markdown(r"""# Bonus Exercises

Congratulations on getting through the core exercises of the first day!""")

    button = st.button("Press me when you're finished! 🙂", on_click = st.balloons)

    st.markdown("""Now, you'll have time to explore some more of these exercises in greater detail. Some suggested exercises:

### Fill out the [feedback form](https://forms.gle/fzp5HbhHjU96NELK8)

This is the most important exercise of all! We're keen to make sure ARENA is as useful and enjoyable as possible, so we're keen to hear about the experience you had with these exercises. Did you like working in Streamlit, or would you prefer to keep everything in Markdown files? Were the exercises appropriately challenging, or too hard / easy? You can let us know here.

### Compare your results from parts 1 and 2

Do the Fourier coefficients that you calculated explicitly in part 1 match the learned Fourier coefficients in part 2? Why, or why not? Does this depend on whether you use quadratic loss or some other loss function (e.g. $L_1$ loss)?

### FFT

The FFT (Fast Fourier Transform) is an algorithm that speeds up the DFT significantly. The DFT is $O(n^2)$, but the FFT is $O(n \log{n})$ when correctly implemented. Can you write a function in Python which implements the DFT? How does it compare in speed to NumPy's built-in DFT functions?

### DFT on PyTorch

Try rewriting your DFT code (and FFT, if you did the exercise above), but using PyTorch tensors rather than NumPy arrays. How does this compare in speed to PyTorch's built-in functions? Can you get another speedup by making your function run on the GPU?

### Fourier series convergence

In the second exercise, you hopefully saw the loss fall over time, down to some lower bound which depended on the function you used and the number of frequencies which were used to approximate the function. Try choosing a fixed function (e.g. $y=x^2$ or the Heaviside step function). Can you see a pattern in the loss lower bound as the number of Fourier terms increases?

You can also try playing around with some different functions, e.g. polynomial / trigonometric / piecewise linear. What features of a function determine the speed of convergence as you add more Fourier terms?
""")

func_list = [section_home, section_fourier, section_nn, section_bonus]

page_list = ["🏠 Home", "1️⃣ Fourier Transforms", "2️⃣ Basic Neural Network", "3️⃣ Bonus Exercises"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()
# for idx, section in enumerate(sections_selectbox):
#     func_list[idx]()
