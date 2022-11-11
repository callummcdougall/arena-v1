import streamlit as st

st.set_page_config(layout="wide")

import platform
is_local = (platform.processor() != "")
rootdir = "" if is_local else "streamlit/ch2/"

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
## 1️⃣ Suggested exercises

Today, we just have a collection of possible exercises that you can try today. The first one is more guided, and closely related to the material you covered yesterday. Many of the rest are based on material from Jacob Hilton's curriculum. They have relatively little guidance, and leave many of the implementation details up to you to decide. Some are more mathematical, others are more focused on engineering and implementation skills. If you're struggling with any of them, you can try speaking with your teammates in the Slack channel, or messaging on `#technical-questions`.

All the exercises here are optional, so you can attempt as many / few as you like. It's highly possible you'll only have time to do one of them, since each of them can take you down quite a long rabbit hole!
""")

def section1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#1-parameter-groups">1. Parameter groups</a></li>
    <li><a class="contents-el" href="#2-benchmark-different-optimizers-and-learning-rates">2. Benchmark different optimizers and learning rates</a></li>
    <li><a class="contents-el" href="#3-noisy-quadratic-model">3. Noisy Quadratic Model</a></li>
    <li><a class="contents-el" href="#4-shampoo">4. Shampoo</a></li>
    <li><a class="contents-el" href="#5-the-colorization-problem">5. The Colorization Problem</a></li>
    <li><a class="contents-el" href="#6-extra-reading">6. Extra reading</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown("""
## 1. Parameter groups

We mentioned parameter groups briefly yesterday, but didn't go into much detail. To elaborate further here: rather than passing a single iterable of parameters into an optimizer, you have the option to pass a list of parameter groups, each one with different hyperparameters. As an example of how this might work:

```python
optim.SGD([
    {'params': model.base.parameters()},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], lr=1e-2, momentum=0.9)
```

The first argument here is a list of dictionaries, with each dictionary defining a separate parameter group. Each should contain a `params` key, which contains an iterable of parameters belonging to this group. The dictionaries may also contain keyword arguments. If a parameter is not specified in a group, PyTorch uses the value passed as a keyword argument. So the example above is equivalent to:

```python
optim.SGD([
    {'params': model.base.parameters(), 'lr': 1e-2, 'momentum': 0.9},
    {'params': model.classifier.parameters(), 'lr': 1e-3, 'momentum': 0.9}
])
```

PyTorch optimisers will store all their params and hyperparams in the `param_groups` attribute, which is a list of dictionaries like the one above, where each one contains *every* hyperparameter rather than just the ones that were specified by the user at initialisation. Optimizers will have this `param_groups` attribute even if they only have one param group - then `param_groups` will just be a list containing a single dictionary.

Your exercise is to rewrite the `SGD` optimizer from yesterday, to use `param_groups`. A few things to keep in mind during this exercise:

* The learning rate must either be specified as a keyword argument, or it must be specified in every group. If it isn't specified as a keyword argument but it is specified in every group, you should raise an error.
    * This isn't true for the other hyperparameters like momentum. They all have default values, and so they don't need to be specified.
* You should add some code to check that no parameters appear in more than one group (PyTorch raises an error if this happens).

```python
class SGD:

    def __init__(self, params, **kwargs):
        '''Implements SGD with momentum.

        Accepts parameters in groups, or an iterable.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        kwargs can contain lr, momentum or weight_decay
        '''
        pass
    
    def zero_grad(self) -> None:
        pass

utils.test_sgd_param_groups(SGD)
```

You can also try to rewrite the learning rate schedulers from yesterday, to work on optimisers which store their params and hyperparams in the `param_groups` attribute (this will just require replacing one line in each of your schedulers).

## 2. Benchmark different optimizers and learning rates

Now that you've learned about different optimizers and learning rates, and you've used `wandb` to run hyperparameter sweeps, you now have the opportunity to combine the two and run your own experiments. A few things which might be interesting to investigate:

* How does the training loss for different optimizers behave when training your ConvNet or ResNets from chapter 0, or your decoder-only transformer from chapter 1?
* It was mentioned yesterday that PyTorch applies weight decay to all parameters equally, rather than only to weights and not to biases. What happens when you run experiments on your ConvNet or ResNet with weight decay varying across weights and biases?
    * Note - you'll need to use **parameter groups** for this task; see exercise 1 above. You can find all the biases by iterating through `model.named_parameters()`, and checking whether the name contains the string `"bias"`.

## 3. Noisy Quadratic Model

As we discussed yesterday, a large bach generall means that the estimate of the gradient is closer to that of the true gradient over the entire dataset (because it is an aggregate of many different datapoints). But empirically, we tend to observe a [**critical batch size**](https://arxiv.org/pdf/1812.06162.pdf), above which training becomes less-data efficient.

The NQM is the second-order Taylor expansion of the loss discussed in the critical batch size paper, and accounts for surprisingly many deep learning phenomena. [This paper](https://arxiv.org/abs/1907.04164) uses this model to explain the effect of curvature and preconditioning on the critical batch size.

You can try running your own set of noisy quadratic model experiments, based on the NQM paper:

* Set up a testbed using the setup from the NQM paper, where the covariance matrix of the gradient and the Hessian are both diagonal. You can use the same defaults for these matrices as in the paper, i.e., diagonal entries of 1, 1/2, 1/3, ... for both (in the paper they go up to 10^4, you can reduce this to 10^3 if experiments are taking too long to run). Implement both SGD with momentum and Adam.
* Create a method for optimizing learning rate schedules. You can either use dynamic programming using equation (3) as in the paper (see footnote on page 7), or a simpler empirical method such as black-box optimization (perhaps with simpler schedule).
* Check that at very small batch sizes, the optimal learning rate scales with batch size as expected: proportional to the batch size for SGD, proportional to the square root of the batch size for Adam.
* Look at the relationship between the batch size and the number of steps to reach a target loss. Study the effects of momentum and using Adam on this relationship.""")

    st.info("""
Note - if you're confused by the concepts of **preconditioned gradient descent** and the **conditioning number** (which come up quite a lot in the NQM paper), you might find [this video](https://www.youtube.com/watch?v=zjzOYL4fhrQ) helpful, as well as the accompanying [Colab notebook](https://colab.research.google.com/drive/1lBu2aprYsOq5wj73Avaf0klRv47GfrOH#scrollTo=eo3DiaJmZ9B9). The latter gives an interactive walkthrough of these concepts, with useful visualisations. Additionally, [this page of lecture notes](https://www.cs.princeton.edu/courses/archive/fall18/cos597G/lecnotes/lecture5.pdf) discusses preconditioning, although it's somewhat more technical than the Colab notebook.

""")

    st.markdown("""
## 4. Shampoo

We briefly mentioned Shampoo yesterday, a special type of structure-aware preconditioning algorithm. Try to implement it, based on algorithm 1 from [this paper](https://arxiv.org/pdf/1802.09568.pdf).""")

    st.image(rootdir + "images/shampoo.png")

    st.markdown("""
You can do this by defining an optimizer just like your `SGD`, `RMSprop` and `Adam` implementations yesterday. Try using your optimizer on Rosenbrock's banana, and on some of the neural networks you've made so far like your ConvNet or ResNet. How does it do compared to the other algorithms?

## 5. The Colorization Problem

This problem was described in the [Distill blog post on momentum](https://distill.pub/2017/momentum/#:~:text=Example%3A%20The%20Colorization%20Problem). Can you implement this problem, and test it out with different optimizers and hyperparameters? What kinds of results do you get?

## 6. Extra reading

If none of the experiments above seem all that exciting to you, then you can read some more about optimisation instead. As well as yesterday's resources, a few you might enjoy are:

* [Deep Double Descent](https://openai.com/blog/deep-double-descent/) - A revision of the classical bias-variance trade-off for deep learning. Further investigation of the phenomenon can be found [here](https://arxiv.org/abs/2002.11328).
* [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) - A well-known counterintuitive result about pruning.
""")

func_list = [section_home, section1]

page_list = ["🏠 Home", "1️⃣ Suggested exercises"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()
# for idx, section in enumerate(sections_selectbox):
#     func_list[idx]()
