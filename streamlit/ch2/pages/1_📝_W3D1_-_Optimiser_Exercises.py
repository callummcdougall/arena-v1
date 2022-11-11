import streamlit as st

import plotly.io as pio
import re
import json

import platform
is_local = (platform.processor() != "")
rootdir = "" if is_local else "streamlit/ch2/"

def read_from_html(filename):
    filename = rootdir + f"images/{filename}.html"
    with open(filename) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}
    return pio.from_json(json.dumps(plotly_json))

def get_fig_dict():
    names = [f"rosenbrock_{i}" for i in range(1, 5)]
    return {name: read_from_html(name) for name in names}

if "fig_dict" not in st.session_state:
    fig_dict = get_fig_dict()
    st.session_state["fig_dict"] = fig_dict
else:
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
## 1️⃣ Optimizers \*\*

Today's exercises will take you through how different optimisation algorithms work (specifically SGD, RMSprop and Adam). You'll write your own optimisers, and use plotting functions to visualise gradient descent on loss landscapes.

## 2️⃣ Learning rate schedulers

You'll also learn about learning rate schedulers, and write some of your own. This section is much shorter than the first, and less important. You should consider it optional.
""")

def section1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#imports">Imports</a></li>
   <li><a class="contents-el" href="#reading">Reading</a></li>
   <li><a class="contents-el" href="#gradient-descent">Gradient Descent</a></li>
   <li><a class="contents-el" href="#stochastic-gradient-descent">Stochastic Gradient Descent</a></li>
   <li><a class="contents-el" href="#batch-size">Batch Size</a></li>
   <li><a class="contents-el" href="#common-themes-in-gradient-based-optimizers">Common Themes in Gradient-Based Optimizers</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#weight-decay">Weight Decay</a></li>
       <li><a class="contents-el" href="#momentum">Momentum</a></li>
   </ul></li>
   <li><a class="contents-el" href="#visualising-optimization-with-rosenbrocks-banana">Visualising Optimization With Rosenbrock's Banana</a></li>
   <li><a class="contents-el" href="#optimize-the-banana">Optimize The Banana</a></li>
   <li><a class="contents-el" href="#build-your-own-optimizers">Build Your Own Optimizers</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#sgd">SGD</a></li>
       <li><a class="contents-el" href="#rmsprop">RMSprop</a></li>
       <li><a class="contents-el" href="#adam">Adam</a></li>
   </ul></li>
   <li><a class="contents-el" href="#plotting-multiple-optimisers">Plotting multiple optimisers</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
## Imports

```python
import torch as t
from torch import nn, optim
import numpy as np

import utils
```

## Reading

Some of these are strongly recommended, while others are optional. If you like, you can jump back to some of these videos while you're going through the material, if you feel like you need to.

* Andrew Ng's video series on gradient descent variants:
    * [Gradient Descent With Momentum](https://www.youtube.com/watch?v=k8fTYJPd3_I)\*\* (9 mins)
    * [RMSProp](https://www.youtube.com/watch?v=_e-LFe_igno)\*\* (7 mins)
    * [Adam](https://www.youtube.com/watch?v=JXQT_vxqwIs&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=23) (7 mins)
* [A Visual Explanation of Gradient Descent Methods](https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c)\*
* [Why Momentum Really Works (distill.pub)](https://distill.pub/2017/momentum/)
""")

    st.markdown(r"""
## Gradient Descent

Yesterday, you implemented backpropagation. Today, we're going to use the gradients produced by backpropagation for optimizing a loss function using gradient descent.

A loss function can be any differentiable function such that we prefer a lower value. To apply gradient descent, we start by initializing the parameters to random values (the details of this are subtle), and then repeatedly compute the gradient of the loss with respect to the model parameters. It [can be proven](https://tutorial.math.lamar.edu/Classes/CalcIII/DirectionalDeriv.aspx) that for an infinitesimal step, moving in the direction of the gradient would increase the loss by the largest amount out of all possible directions.

We actually want to decrease the loss, so we subtract the gradient to go in the opposite direction. Taking infinitesimal steps is no good, so we pick some learning rate $\lambda$ (also called the step size) and scale our step by that amount to obtain the update rule for gradient descent:

$$ \theta_t \leftarrow \theta_{t-1} - \lambda \nabla L(\theta_{t-1}) $$

We know that an infinitesimal step will decrease the loss, but a finite step will only do so if the loss function is linear enough in the neighbourhood of the current parameters. If the loss function is too curved, we might actually increase our loss.

The biggest advantage of this algorithm is that for N bytes of parameters, you only need N additional bytes of memory to store the gradients, which are of the same shape as the parameters. GPU memory is very limited, so this is an extremely relevant consideration. The amount of computation needed is also minimal: one multiply and one add per parameter.

The biggest disadvantage is that we're completely ignoring the curvature of the loss function, not captured by the gradient consisting of partial derivatives. Intuitively, we can take a larger step if the loss function is flat in some direction or a smaller step if it is very curved. Generally, you could represent this by some matrix P that pre-multiplies the gradients to rescale them to account for the curvature. P is called a preconditioner, and gradient descent is equivalent to approximating P by an identity matrix, which is a very bad approximation.

Most competing optimizers can be interpreted as trying to do something more sensible for P, subject to the constraint that GPU memory is at a premium. In particular, constructing P explicitly is infeasible, since it's an $N \times N$ matrix and N can be hundreds of billions. One idea is to use a diagonal P, which only requires N additional memory. An example of a more sophisticated scheme is [Shampoo](https://arxiv.org/pdf/1802.09568.pdf).
""")

    st.info("""The algorithm is called **Shampoo** because you put shampoo on your hair before using conditioner, and this method is a pre-conditioner.
    
If you take away just one thing from this entire curriculum, please don't let it be this.""")

    st.markdown("""
## Stochastic Gradient Descent

The terms gradient descent and SGD are used loosely in deep learning. To be technical, there are three variations:

- Batch gradient descent - the loss function is the loss over the entire dataset. This requires too much computation unless the dataset is small, so it is rarely used in deep learning.
- Stochastic gradient descent - the loss function is the loss on a randomly selected example. Any particular loss may be completely in the wrong direction of the loss on the entire dataset, but in expectation it's in the right direction. This has some nice properties but doesn't parallelize well, so it is rarely used in deep learning.
- Mini-batch gradient descent - the loss function is the loss on a batch of examples of size `batch_size`. This is the standard in deep learning.

The class `torch.SGD` can be used for any of these by varying the number of examples passed in. We will be using only mini-batch gradient descent in this course.

## Batch Size

In addition to choosing a learning rate or learning rate schedule, we need to choose the batch size or batch size schedule as well. Intuitively, using a larger batch means that the estimate of the gradient is closer to that of the true gradient over the entire dataset, but this requires more compute. Each element of the batch can be computed in parallel so with sufficient compute, one can increase the batch size without increasing wall-clock time. For small-scale experiments, a good heuristic is thus "fill up all of your GPU memory".

At a larger scale, we would expect diminishing returns of increasing the batch size, but empirically it's worse than that - a batch size that is too large generalizes more poorly in many scenarios. The intuition that a closer approximation to the true gradient is always better is therefore incorrect. See [this paper](https://arxiv.org/pdf/1706.02677.pdf) for one discussion of this.

For a batch size schedule, most commonly you'll see batch sizes increase over the course of training. The intuition is that a rough estimate of the proper direction is good enough early in training, but later in training it's important to preserve our progress and not "bounce around" too much.

You will commonly see batch sizes that are a multiple of 32. One motivation for this is that when using CUDA, threads are grouped into "warps" of 32 threads which execute the same instructions in parallel. So a batch size of 64 would allow two warps to be fully utilized, whereas a size of 65 would require waiting for a third warp to finish. As batch sizes become larger, this wastage becomes less important.

Powers of two are also common - the idea here is that work can be recursively divided up among different GPUs or within a GPU. For example, a matrix multiplication can be expressed by recursively dividing each matrix into four equal blocks and performing eight smaller matrix multiplications between the blocks.

In tomorrow's exercises, you'll have the option to expore batch sizes in more detail.

## Common Themes in Gradient-Based Optimizers

### Weight Decay

Weight decay means that on each iteration, in addition to a regular step, we also shrink each parameter very slightly towards 0 by multiplying a scaling factor close to 1, e.g. 0.9999. Empirically, this seems to help but there are no proofs that apply to deep neural networks.

In the case of linear regression, weight decay is mathematically equivalent to having a prior that each parameter is Gaussian distributed - in other words it's very unlikely that the true parameter values are very positive or very negative. This is an example of "**inductive bias**" - we make an assumption that helps us in the case where it's justified, and hurts us in the case where it's not justified.

For a `Linear` layer, it's common practice to apply weight decay only to the weight and not the bias. It's also common to not apply weight decay to the parameters of a batch normalization layer. Again, there is empirical evidence (such as [Jai et al 2018](https://arxiv.org/pdf/1807.11205.pdf)) and there are heuristic arguments to justify these choices, but no rigorous proofs. Note that PyTorch will implement weight decay on the weights *and* biases of linear layers by default - see the bonus exercises tomorrow for more on this.

### Momentum

Momentum means that the step includes a term proportional to a moving average of past gradients. [Distill.pub](https://distill.pub/2017/momentum/) has a great article on momentum, which you should definitely read if you have time. Don't worry if you don't understand all of it; skimming parts of it can be very informative. For instance, the first half discusses the **conditioning number** (a very important concept to understand in optimisation), and concludes by giving an intuitive argument for why we generally set the momentum parameter close to 1 for ill-conditioned problems (those with a very large conditioning number).

## Visualising Optimization With Rosenbrock's Banana

"Rosenbrock's Banana" is a (relatively) famous function that has a simple equation but is challenging to optimize because of the shape of the loss landscape.

We've provided you with a function to calculate Rosenbrock's Banana, and another one to plot arbitrary functions. You can see them both below:

```python
def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1

x_range = [-2, 2]
y_range = [-1, 3]
fig = utils.plot_fn(rosenbrocks_banana, x_range, y_range, log_scale=True)
```

Your output should look like:
""")

    st.plotly_chart(fig_dict["rosenbrock_1"], use_container_width=True)

    with st.expander("Question - where is the minimum of this function?"):
        st.markdown("""
The first term is minimised when `x=a` and the second term when `y=x**2`. So we deduce that the minimum is at `(a, a**2)`. When `a=1`, this gives us the minimum `(1, 1)`.

You can pass the extra argument `show_min=True` to all plotting functions, to indicate the minimum.
""")

    st.markdown("""
## Optimize The Banana

Implement the `opt_fn` function using `torch.optim.SGD`. Starting from `(-1.5, 2.5)`, run your function and add the resulting trajectory of `(x, y)` pairs to your contour plot. Did it find the minimum? Play with the learning rate and momentum a bit and see how close you can get within 100 iterations.

```python
def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    assert xy.requires_grad
    pass
```
""")

    with st.expander("Help - I'm not sure if my `opt_banana` is implemented properly."):
        st.markdown("With a learning rate of `0.001` and momentum of `0.98`, my SGD was able to reach `[ 1.0234,  1.1983]` after 100 iterations.")

    with st.expander("Help - all my (x, y) points are the same."):
        st.markdown("""This is probably because you've stored your `xy` values in a list, so they change each time you perform a gradient descent step. 

Instead, try creating a tensor of zeros to hold them, and fill in that tensor using `xys[i] = xy.detach()` at each step.""")

    with st.expander("Help - I'm getting 'Can't call numpy() on Tensor that requires grad'."):
        st.markdown("""
This is a protective mechanism built into PyTorch. The idea is that once you convert your Tensor to NumPy, PyTorch can no longer track gradients, but you might not understand this and expect backprop to work on NumPy arrays.

All you need to do to convince PyTorch you're a responsible adult is to call detach() on the tensor first, which returns a view that does not require grad and isn't part of the computation graph.
""")

    st.markdown("""
We've also provided you with a function `plot_optimisation_sgd` to plot the steps in your optimisation algorithm. It can be run like this:

```python
xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]

fig = utils.plot_optimization_sgd(opt_fn_with_sgd, rosenbrocks_banana, xy, x_range, y_range, lr=0.001, momentum=0.98, show_min=True)

fig.show()
```

Hopefully, you should see output like this:
""")

    st.plotly_chart(fig_dict["rosenbrock_2"], use_container_width=True)

    st.markdown("""
## Build Your Own Optimizers

Now let's build our own drop-in replacement for these three classes from `torch.optim`. The documentation pages for these algorithms have pseudocode you can use to implement your step method.

""")
    st.info("""
**A warning regarding in-place operations**

Be careful with expressions like `x = x + y` and `x += y`. They are NOT equivalent in Python.

- The first one allocates a new `Tensor` of the appropriate size and adds `x` and `y` to it, then rebinds `x` to point to the new variable. The original `x` is not modified.
- The second one modifies the storage referred to by `x` to contain the sum of `x` and `y` - it is an "in-place" operation.
    - Another way to write the in-place operation is `x.add_(y)` (the trailing underscore indicates an in-place operation).
    - A third way to write the in-place operation is `torch.add(x, y, out=x)`.
- This is rather subtle, so make sure you are clear on the difference. This isn't specific to PyTorch; the built-in Python `list` follows similar behavior: `x = x + y` allocates a new list, while `x += y` is equivalent to `x.extend(y)`.

The tricky thing that happens here is that both the optimizer and the `Module` in your model have a reference to the same `Parameter` instance. 
""")

    with st.expander("Question - should we use in-place operations in our optimizer?"):
        st.markdown("""
You MUST use in-place operations in your optimizer because we want the model to see the change to the Parameter's storage on the next forward pass. If your optimizer allocates a new tensor, the model won't know anything about the new tensor and will continue to use the old, unmodified version.

Note, this observation specifically refers to the parameters. When you're updating non-parameter variables that you're tracking, you should be careful not to accidentally use an in-place operation where you shouldn't!")""")

    st.markdown("""### More Tips

- The provided `params` might be a generator, in which case you can only iterate over it once before the generator is exhausted. Copy it into a `list` to be able to iterate over it repeatedly
- Your step function shouldn't modify the gradients. Use the `with torch.inference_mode():` context for this. Fun fact: you can instead use `@torch.inference_mode()` (note the preceding `@`) as a method decorator to do the same thing.
- If you create any new tensors, they should be on the same device as the corresponding parameter. Use `torch.zeros_like()` or similar for this.
- Be careful not to mix up `Parameter` and `Tensor` types in this step.
- The actual PyTorch implementations have an additional feature called parameter groups where you can specify different hyperparameters for each group of parameters. You can ignore this for now; we'll come back to it tomorrow.

Note, the configurations used during testing will start simple (e.g. all parameters set to zero except `lr`) and gradually move to more complicated ones. This will help you track exactly where in your model the error is coming from.

You should also fill in the default PyTorch keyword arguments, where appropriate.

### SGD

```python
class SGD:
    params: list

    def __init__(self, params: Iterable[t.nn.parameter.Parameter], lr: float, momentum: float, weight_decay: float):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        '''
        pass

    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        pass

    def __repr__(self) -> str:
        # Should return something reasonable here, e.g. "SGD(lr=lr, ...)"
        pass

utils.test_sgd(SGD)
```

If you've having trouble, you can use the following process when implementing your optimisers:

1. Take the pseudocode from the PyTorch documentation page, and write out the "simple version", i.e. without all of the extra variables which you won't need. (It's good practice to be able to parse pseudocode and figure out what it actually means - during the course we'll be doing a lot more of "transcribing instructions / formulae from paper into code"). You'll want pen and paper for this!

2. Figure out which extra variables you'll need to track within your class.

3. Implement the `step` function using these variables.

You can click on the expander below to see what the first two steps look like for the case of SGD (try and have a go at each step before you look).
""")

    with st.expander("STEP 1"):
        st.markdown(r"""
In the SGD pseudocode, you'll first notice that we can remove the nesterov section, i.e. we always do $g_t \leftarrow \boldsymbol{b}_t$. Then, we can actually remove the variable $\boldsymbol{b_t}$ altogether (because we only needed it to track values while implementing nesterov). Lastly, we have `maximize=False` and `dampening=0`, which allows us to further simplify. So we get the simplified pseudocode:

$
\text {for } t=1 \text { to } \ldots \text { do } \\
\quad g_t \leftarrow \nabla_\theta f_t\left(\theta_{t-1}\right) \\
\quad \text {if } \lambda \neq 0 \\
\quad\quad g_t \leftarrow g_t+\lambda \theta_{t-1} \\
\quad \text {if } \mu \neq 0 \text{ and } t>1 \\
\quad\quad g_t \leftarrow \mu g_{t-1} + g_t \\
\quad \theta_t \leftarrow \theta_{t-1} - \gamma g_t
$

Note - you might find it helpful to name your variables in the `__init__` step in line with their definitions in the pseudocode, e.g. `self.mu = momentum`. This will make it easier to implement the `step` function.
""")

    with st.expander("STEP 2"):
        st.markdown(r"""
In the formula from STEP 1, $\theta_t$ represents the parameters themselves, and $g_t$ represents variables which we need to keep track of in order to implement momentum. We need to track $g_t$ in our model, e.g. using a line like:

```python
self.gs = [t.zeros_like(p) for p in self.params]
```

We also need to track the variable $t$, because the behavour is different when $t=0$. (Technically we could just as easily not do this, because the behaviour when $t=0$ is just the same as the behaviour when $g_t=0$ and $t>0$. But I've left $t$ in my solutions, to make it more obvious how the `SGD.step` function corrsponds to the pseudocode shown in STEP 1.

Now, you should be in a good position to attempt the third step: applying SGD in the `step` function, using this algorithm and these tracked variables.
""")


    st.markdown("""
### RMSprop

Once you've implemented SGD, you should do RMSprop in a similar way. Although the pseudocode is more complicated and there are more variables you'll have to track, there is no big conceptual difference between the task for RMSprop and SGD.

If you want to better understand why RMSprop works, then you can return to some of the readings at the top of this page.

```python
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
        '''
        pass

    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        pass

    def __repr__(self) -> str:
        pass
    


utils.test_rmsprop(RMSprop)
```""")

    st.markdown(r"""
### Adam

Finally, you'll do the same for Adam. This is a very popular optimizer in deep learning, which empirically often outperforms most others. It combines the heuristics of both momentum (via the $\beta_1$ parameter), and RMSprop's handling of noisy data by dividing by the $l_2$ norm of gradients (via the $\beta_2$ parameter).

```python
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        '''
        pass

    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        pass

    def __repr__(self) -> str:
        pass

utils.test_adam(Adam)
```

## Plotting multiple optimisers

Finally, we've provided some code which should allow you to plot more than one of your optimisers at once.

First, you should fill in this function, which will be just like your `opt_fn_with_sgd` from earlier, except that it works when passed an arbitrary optimizer (from the ones that you've defined).

```python
def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_kwargs, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    assert xy.requires_grad
    pass
```

Once you've implemented this function, you can use `utils.plot_optimization` to create plots of multiple different optimizers at once. An example of how this should work can be found below. The argument `optimizers` should be a list of tuples `(optimizer_class, optimizer_kwargs)` which will get passed into `opt_fn`.

```python
xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]
optimizers = [
    (solution.SGD, dict(lr=1e-3, momentum=0.98)),
    (solution.SGD, dict(lr=5e-4, momentum=0.98)),
]

fig = utils.plot_optimization(opt_fn, fn, xy, optimizers, x_range, y_range)

fig.show()
```
""")

    st.plotly_chart(fig_dict["rosenbrock_3"], use_container_width=True)

    st.markdown("""
You can try and play around with a few optimisers. Do Adam and RMSprop do well on this function? Why / why not? Can you find some other functions where they do better / worse, and plot those?
""")

def section2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#why-learning-rate-schedules">Why learning rate schedulers?</a></li>
    <li><a class="contents-el" href="#writing-your-own-schedulers">Writing your own schedulers</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#exponentiallr">ExponentialLR</a></li>
        <li><a class="contents-el" href="#steplr">StepLR</a></li>
        <li><a class="contents-el" href="#multisteplr">MultiStepLR</a></li>
    <li><a class="contents-el" href="#other-schedulers">Other schedulers</a></li>
    <li><a class="contents-el" href="#plotting-lr-schedulers">Plotting LR schedulers</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""

## Why learning rate schedulers?

As the last set of exercises today, we'll learn about learning rate schedules. These are ways in which you can control the learning rate of your optimizer over time. PyTorch allows you to do this via the [`torch.optim.lr_scheduler`](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) library. For example, consider the following code:

```python
model = [nn.Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = optim.SGD(model, 0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

The scheduler takes the optimizer as first argument (since it's the optimizer's parameters that the scheduler will be updating, much like the optimizer updates the model's parameters). The other arguments are keyword arguments which control the behaviour of the scheduler.

Within each epoch, we have the standard training loop that you're probably used to. But at the end of each loop, we now have the extra line `scheduler.step()`. This updates the learning rate of the optimizer according to the scheduler's ruleset.

Why is it worth knowing about schedulers? Partly because varying the learning rate over time can lead to improved performance, even for gradient descent algorithms which are already adaptive. In general, it is common to start out with a high learning rate (as your model is randomly initialised) and then decay it over time (as you get closer to optimal performance, and a large learning rate would lead to instability). Also, you'll often need to get these details right when implementing papers. For instance, we'l have to deal with learning rate adjustment during the RL chapter, and while building diffusion models (for those of you who choose the 'Modelling Objectives' track).

We will have you implement three learning rate algorithms: `ExponentialLR`, `StepLR` and `MultiStepLR`. Since the basic principles behind learning rate schedulers generalise pretty easily, you're welcome to move on from this section once you feel like you have a good grasp on things.

One last note = when implementing these functions, **you should make them work with the optimizers you've already built**. This means that your learning rate will be stored as `optimizer.lr` (or whatever name you chose for it). This isn't actually the case for optimizers in `torch.optim`, since learning rates and other parameters are actually stored in `optimizer.param_groups` (which we'll cover tomorrow).

## Writing your own schedulers

### ExponentialLR

This scheduler updates the learning rate each time `step` is called, by multiplying it by the factor `gamma`.

```python
class ExponentialLR():
    def __init__(self, optimizer, gamma):
        '''Implements ExponentialLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
        '''
        pass

    def step(self):
        pass

    def __repr__(self):
        pass

test_ExponentialLR(ExponentialLR, SGD)
```

### StepLR

This is just like `ExponentialLR`, except that it only updates every `step_size` steps, rather than every step.

```python
class StepLR():
    def __init__(self, optimizer, step_size, gamma=0.1):
        '''Implements StepLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        '''
        pass

    def step(self):
        pass

    def __repr__(self):
        pass

utils.test_StepLR(StepLR, SGD)
```

### MultiStepLR

This is just like `ExponentialLR`, except that it only updates if the number of steps it's taken is in the list `milestones`.

```python
class MultiStepLR():
    def __init__(self, optimizer, milestones, gamma=0.1):
        '''Implements MultiStepLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html
        '''
        pass

    def step(self):
        pass

    def __repr__(self):
        pass

utils.test_MultiStepLR(MultiStepLR, SGD)
```

## Other schedulers

There are more advanced learning rate schedulers available in PyTorch. For example, there are [cyclic learning rates](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR), which are based on the core intuition that an optimal learning rate range exists, and we should vary the learning rate in a disciplined way in order to make sure you spend time in that range.""")

    st.image(rootdir + "images/lr.png")

    st.markdown("""
There are also more complicated dependencies you can use - for instance, [`ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau) will reduce the learning rate once a metric has stopped improving. 

## Plotting LR schedulers

Finally, you can fill in this function which optimises over a given function, with optimizer *and learning rate scheduler* specified. You should have the learning rate scheduler step at the end of each gradient descent step (rather than at the end of each epoch, which is the normal convention when using schedulers). 

```python
def opt_fn_with_scheduler(
    fn: Callable, 
    xy: t.Tensor, 
    optimizer_class, 
    optimizer_kwargs, 
    scheduler_class = None, 
    scheduler_kwargs = dict(), 
    n_iters: int = 100
):
    '''Optimize the a given function starting from the specified point.

    scheduler_class: one of the schedulers you've defined, either ExponentialLR, StepLR or MultiStepLR
    scheduler_kwargs: keyword arguments passed to your optimiser (e.g. gamma)
    '''
    assert xy.requires_grad
    pass
```

Finally, we've provided you some functions you can use to plot the effect of different optimisation algorithms *and* learning rate schedules. You can use it as follows:

```python
xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]
optimizers = [
    (solution.SGD, dict(lr=1e-3, momentum=0.98)),
    (solution.SGD, dict(lr=1e-3, momentum=0.98)),
]
schedulers = [
    (), # Empty list stands for no scheduler
    (solution.ExponentialLR, dict(gamma=0.99)),
]

fig = utils.plot_optimization_with_schedulers(solution.opt_fn_with_scheduler, fn, xy, optimizers, schedulers, x_range, y_range, show_min=True)

fig.show()
```
""")

    st.plotly_chart(fig_dict["rosenbrock_4"], use_container_width=True)

    st.markdown("""
How close can you get to the optimum within 100 steps, starting from (-1.5, 2.5)? Share screenshots of your best runs on Slack!
""")
    

func_list = [section_home, section1, section2]

page_list = ["🏠 Home", "1️⃣ Optimizers", "2️⃣ Learning rate schedulers"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()

# for idx, section in enumerate(sections_selectbox):
#     func_list[idx]()


# Main exercises:

# function to run sgd on a function, and plot it

# implementing your own sgd variants
# implementing learning rates (parameter groups are assumed)

# bonus

# bayesian optimisation of hyperparameters - implement your own? there's a post on this in distil
# implement parameter groups
# empirical model of large-batch training: https://github.com/hoangcuong2011/Good-Papers/blob/master/An%20Empirical%20Model%20of%20Large-Batch%20Training.md
# weight decay: compare result of weight decay on weights, and weight decay on bias. PyTorch automatically applies it to weights and biases.
    # hint: to get all the biases in param groups, use the following code:

    # param_lists = {"bias": [], "no_bias": []}
    # for (name, param) in model.named_parameters():
    #   if "bias" in name:
    #       param_lists["bias"].append(param)
    #   else:
    #       param_lists["no_bias"].append(param)

    # then use param groups
# other things on Jacob's curriculum (remember they're on Conor's solns document!)





