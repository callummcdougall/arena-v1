import streamlit as st

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
    st.markdown(r"""
# Weights and Biases

Hopefully, last week you were able to successfully implement a transformer last week (or get pretty close!). If you haven't done that yet, then this should be your first priority going forwards with this week.

Today, we'll look at methods for choosing hyperparameters effectively. You'll learn how to use **Weights and Biases**, a useful tool for hyperparameter search, which should allow you to tune your own transformer model by the end of today's exercises.

The exercises themselves will be based on your ResNet implementations from week 0 (although the priciples should carry over to your transformer models in exactly the same way). If you weren't able to get your resnet working in week 0, you can just use the solution in the GitHub repo (alternatively, this might be another good opportunity to try and get yours working!).
""")

def section_wandb():
    st.markdown("""
## Imports

You may have to install `wandb`.

```python
import torch as t
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fancy_einsum import einsum
from typing import Union, Optional, Callable
import numpy as np
from einops import rearrange
from tqdm.notebook import tqdm_notebook
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import wandb
import utils

device = "cuda" if t.cuda.is_available() else "cpu"
```

## CIFAR10

The benchmark we'll be training on is [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), which consists of 60000 32x32 colour images in 10 different classes. Don't peek at what other people online have done for CIFAR10 (it's a common benchmark), because the point is to develop your own process by which you can figure out how to improve your model. Just reading the results of someone else would prevent you from learning how to get the answers. To get an idea of what's possible: using one V100 and a modified ResNet, one entry in the DAWNBench competition was able to achieve 94% test accuracy in 24 epochs and 76 seconds. 94% is approximately [human level performance](http://karpathy.github.io/2011/04/27/manually-classifying-cifar10/).

Below is some boilerplate code for downloading and transforming `CIFAR10` data (this shouldn't take more than a minute to run the first time). There are a few differences between this and our code last week - for instance, we omit the `transforms.Resize` from our `transform` object, because CIFAR10 data is already the correct size (unlike the sample images from last week).

```python
cifar_mean = [0.485, 0.456, 0.406]
cifar_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

utils.show_cifar_images(trainset, rows=3, cols=5)
```

We have also provided a basic training & testing loop, almost identical to the one you used in W0D3. This one doesn't use `wandb` at all, although it does plot the train loss and test accuracy when the function finishes running. You should run this function to verify your model is working, and that the loss is going down. Also, make sure you understand what each part of this function is doing.
""")

    with st.expander("TRAIN FUNCTION - SIMPLE"):

        st.markdown("""
```python
def train(trainset, testset, epochs: int, loss_fn: Callable, batch_size: int, lr: float) -> tuple[list, list]:

    model = ResNet34().to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    
    loss_list = []
    accuracy_list = []

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            accuracy_list.append(accuracy/total)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = "./w0d3_resnet.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)

    utils.plot_results(loss_list, accuracy_list)
    return loss_list, accuracy_list
    
epochs = 1
loss_fn = nn.CrossEntropyLoss()
batch_size = 128
lr = 0.001
    
loss_list, accuracy_list = train(trainset, testset, epochs, loss_fn, batch_size, lr)
```
""")

    st.markdown("""
## What is Weights and Biases?

Weights and Biases is a cloud service that allows you to log data from experiments. Your logged data is shown in graphs during training, and you can easily compare logs across different runs. It also allows you to run **sweeps**, where you can specifiy a distribution over hyperparameters and then start a sequence of test runs which use hyperparameters sampled from this distribution.

The way you use weights and biases is pretty simple. You take a normal training loop, and add a bit of extra code to it. The only functions you'll need to use today are:

* `wandb.init`, which starts a new run to track and log to W&B
* `wandb.watch`, which hooks into your model to track parameters and gradients
* `wandb.log`, which logs metrics and media over time within your training loop
* `wandb.save`, which saves the details of your run
* `wandb.sweep` and `wandb.agent`, which are used to run hyperparameter sweeps

You should visit the [Weights and Biases homepage](https://wandb.ai/home), and create your own user. You will also have to login the first time you run `wandb` code (this can be done by running `wandb login` in whichever terminal you are using).

## Logging runs with `wandb`

The most basic way you can use `wandb` is by logging variables during your run. This removes the need for excessive printing of output. Below is an example training loop which does this.
""")

    with st.expander("TRAIN FUNCTION - WANDB LOGGING"):
        st.markdown("""
```python
def train(trainset, testset, epochs: int, loss_fn: Callable, batch_size: int, lr: float) -> None:

    config_dict = {
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }
    wandb.init(project="w2d1_resnet", config=config_dict)

    model = ResNet34().to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    examples_seen = 0
    start_time = time.time()

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)

        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

train(trainset, testset, epochs, loss_fn, batch_size, lr)
```
""")

    st.markdown("""
When you run this function, it should give you a url which you can click on to see a graph of parameters plotted over time. You can also switch to a tabular view, which will let you compare multiple runs.

Some notes on the parts of this function which have been changed:

#### `wandb.init`

This starts a new run to track and log to Weights and Biases. The `project` keyword allows you to group all your runs in a single directory, and the `config` keyword accepts a dictionary of hyperparameters which will be logged to wandb (you can see these in your table when you compare different runs).

`wandb.init` must always be the very first `wandb` function that gets called.

#### `wandb.watch`

This hooks into your model to track parameters and gradients. You should be able to see graphs of your parameter and gradient values (`log="all"` means that both of these are tracked; the other options being `log="gradients"` and `log="parameters"`).

#### `wandb.log`

This logs metrics and media over time within your training loop. The two arguments it takes here are `data` (a dictionary of values which should be logged; you will be able to see a graph for each of these parameters on Weights and Biases) and `step` (which will be the x-axis of the graphs). If `step` is not provided, by default `wandb` will increment step once each time it logs, so `step` will actually correspond to the batch number. In the code above, `step` corresponds to the total number of examples seen.

#### `wandb.save`

This saves the details of your run. You can view your saved models by navigating to a run page, clicking on the `Files` tab, then clicking on your model file.

You should try runing this function a couple of times with some different hyperparameters, and get an idea for how it works.

## Hyperparameter search

One way to do hyperparameter search is to choose a set of values for each hyperparameter, and then search all combinations of those specific values. This is called **grid search**. The values don't need to be evenly spaced and you can incorporate any knowledge you have about plausible values from similar problems to choose the set of values. Searching the product of sets takes exponential time, so is really only feasible if there are a small number of hyperparameters. I would recommend forgetting about grid search if you have more than 3 hyperparameters, which in deep learning is "always".

A much better idea is for each hyperparameter, decide on a sampling distribution and then on each trial just sample a random value from that distribution. This is called **random search** and back in 2012, you could get a [publication](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) for this. The diagram below shows the main reason that random search outperforms grid search. Empirically, some hyperparameters matter more than others, and random search benefits from having tried more distinct values in the important dimensions, increasing the chances of finding a "peak" between the grid points.
""")

    st.image("ch1/images/grid_vs_random.png")

    st.markdown("""
It's worth noting that both of these searches are vastly less efficient than gradient descent at finding optima - imagine if you could only train neural networks by randomly initializing them and checking the loss! Either of these search methods without a dose of human (or eventually AI) judgement is just a great way to turn electricity into a bunch of models that don't perform very well.
""")

    st.markdown("""
## Running hyperparameter sweeps with `wandb`

Now we've come to one of the most impressive features of `wandb` - being able to perform hyperparameter sweeps. Below is a final function which implements hyperparameter sweeps.
""")

    with st.expander("TRAIN FUNCTION - WANDB SWEEP"):
        st.markdown("""
```python
def train() -> None:

    wandb.init()

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    model = ResNet34().to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    examples_seen = 0
    start_time = time.time()

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)


        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

sweep_config = {
    'method': 'random',
    'name': 'w2d1_resnet_sweep_2',
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': 
    {
        'batch_size': {'values': [64, 128, 256]},
        'epochs': {'min': 1, 'max': 3},
        'lr': {'max': 0.1, 'min': 0.0001, 'distribution': 'log_uniform_values'}
     }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='w2d1_resnet')

wandb.agent(sweep_id=sweep_id, function=train, count=2)
```
""")

    st.markdown("""
There's a lot of extra stuff here, but in fact the core concepts are relatively straightforward. We'll go through the new code line by line.

Firstly, note that we've kept everything the same in our `train` function except for the code at the start. The function now doesn't take any arguments (`trainset`, `testset` and `loss_fn` are used as global variables), and the hyperparameters `epochs`, `batch_size` and `lr` are now defined from `wandb.config` rather than being passed into `wandb.init` via the `config` argument.

Most of the extra code comes at the end. First, let's look at `sweep_config`. This dictionary provides all the information `wandb` needs in order to conduct hyperparameter search. The important keys are:

* `method`, which determines how the hyperparameters are searched for.
    * `random` just means random search, as described above.
    * Other options are `grid` (also described above) and `bayes` (which is a smart way of searching for parameters that adjusts in the direction of expected metric improvement based on a Gaussian model).
* `name`, which is just the name of your sweep in Weights and Biases.
* `metric`, which is a dictionary of two keys: `name` (what we want to optimise) and `goal` (the direction we want to optimise it in). 
    * Note that `name` must be something which is logged by our model in the training loop (in this case, `'test_accuracy'`).
    * You can also be clever with the metrics that you maximise: for instance:
        * Minimising training time, by having your training loop terminate early when the loss reaches some threshold (although you'll have to be careful here, e.g. in cases where your loss never reaches the threshold).
        * Optimising some function of multiple metrics, e.g. a weighted sum.
* `parameters`, which is a dictionary with items of the form `hyperparameter_name: search_method`. This determines which hyperparameters are searched over, and how the search is conducted.
    * There are several ways to specify hyperparameter search in each `search_method`. You can read more [here](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration).
    * The simplest search methods are `values` (choose uniformly from a list of values).
        * This can also be combined with `probabilities`, which should be a list specifying the probability of selecting each element from `values`.
    * You can also specify `min` and `max`, which causes wandb to choose uniformly, either from a discrete or continuous uniform distribution depending on whether the values for `min` and `max` are integers or floats.
    * You can also pass the argument `distribution`, which gives you more control over how the random values are selected. For instance, `log_uniform_values` returns a value `X` between `min` and `max` s.t. `log(X)` is uniformly distributed between `log(min)` and `log(max)`.
        * (Question - can you see why a log uniform distribution for `lr` makes more sense than a uniform distribution?)
""")

    with st.expander("Note on using YAML files (optional)"):
        st.markdown("""
Rather than using a dictionary, you can alternatively store the `sweep_config` data in a YAML file if you prefer. You will then be able to run a sweep via the following terminal command:

```
wandb sweep sweep_config.yaml

wandb agent <SWEEP_ID>
```

where `SWEEP_ID` is the value returned from the first terminal command. You will also need to add another line to the YAML file, specifying the program to be run. For instance, your YAML file might start like this:

```yaml
program: train.py
method: random
metric:
    name: test_accuracy
    goal: maximize
```
""")

    st.markdown("""

The final two lines above return a `sweep_id`, and then use that ID to run a sweep. Note that `wandb.agent`'s arguments include a named function (this is why it was important for our `train` function not to take any arguments), and `count` (which determines how many sweeps will be run before the process terminates). 

When you run the code above, you will be given a url called **Sweep page**, in output that will look like:

```
Sweep page: https://wandb.ai/<WANDB-USERNAME>/<PROJECT-NAME>/<SWEEP_ID>
```

This URL will bring you to a page comparing each of your sweeps. You'll be able to see overlaid graphs of each of their training loss and test accuracy, as well as a bunch of other cool things like:

* Bar charts of the [importance](https://docs.wandb.ai/ref/app/features/panels/parameter-importance) (and correlation) of each hyperparameter wrt the target metric. Note that only looking at the correlation could be misleading - something can have a correlation of 1, but still have a very small effect on the metric.
* A [parallel coordinates plot](https://docs.wandb.ai/ref/app/features/panels/parallel-coordinates), which summarises the relationship between the hyperparameters in your config and the model metric you're optimising.

---

`wandb` is an incredibly useful tool when training models, and you should find yourself using it a fair amount throughout this programme. You can always return to this page of exercises if you forget how any part of it works.
""")


func_list = [section_home, section_wandb]

page_list = ["🏠 Home", "1️⃣ Weights and Biases"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()
