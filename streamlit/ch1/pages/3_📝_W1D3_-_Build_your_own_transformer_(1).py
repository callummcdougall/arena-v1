import streamlit as st

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

def section_home():
    st.markdown("""
# Building your own transformer (1)

Yesterday you should have started going through the steps of creating a decoder-only transformer. Specifically, you should have:

* Implemented a positional encoding function
* Written a single-head attention block, with masking

Today, you'll take the next steps to assemble a full transformer. This includes:

* Writing a multihead attention block
    * This should use your functions from yesterday, but you'll be creating a multi-head attention block rather than assuming the matrices $Q$, $K$ and $V$ correspond to single heads.
* Assembling the full decoder-only transformer architecture

---

This is meant to be a very challenging task (and like yesterday we haven't included test functions), so I'm putting in two office hour sessions tomorrow: from 8-10am and 4-5.30pm BST (these have been added to the ARENA Google Calendar).

---

If you find yourself extremely stuck at any point during today's implementation, here are some things you can do. Try and do the first ones on this list before you do the latter ones; struggling and persevering with bugs is a big part of ML engineering!

* Read back over Monday's material
    * Two important resources here are [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and [Language Modelling with Transformers](https://docs.google.com/document/d/1j3EqnPnlg2g2z8fjst4arbZ_hLg_MgE0yFwdSoI237I/edit#heading=h.icw2n6iwy2of)
    * The paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) also gives some useful information
* Speak to your coworkers
    * You can share screen, and try and work through bugs together
    * Sometimes, just [rubber duck debugging](https://en.wikipedia.org/wiki/Rubber_duck_debugging) helps!
* Speak to me (Callum)
    * Either during one of my two office hours today, or just by pinging me a message on Slack
    * I can look over your code, and check for any bugs
* Read [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding)
    * We purposefully didn't inlude this in the recommended reading because we don't want people to just copy the implementation here, also some of it (like the encoder/decoder parts) you don't need, because you're implementing a decoder-only model
    * But it might be useful to get an idea for how certain blocks can be written
""")
 
def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#using-dataclasses">Using <code>dataclasses</code></a></li>
    <li><a class="contents-el" href="#reviewing-positional-encoding">Reviewing Positional Encoding</a></li>
    <li><a class="contents-el" href="#decoderblock-and-decoderonlytransformer">DecoderBlock and DecoderOnlyTransformer</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""

# Multihead Attention""")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
First, you should write a function `multihead_masked_attention`, which takes arguments `Q, K, V` as well as another argument `num_heads`, and performs a multihead attention calculation. This should be the same as the masked attention function you wrote yesterday, except that rather than having shape `(batch, seq_len, d_k)` (or `d_v`), the matrices will have shape `(batch, seq_len, n_heads * d_k)`, where `n_heads` is the number of heads.

Note - it is common practice to assume `d_k = d_v` (from now on, we will refer to this value as `headsize`).

Another note - once you've computed `Q` and `K`, you will have to rerrange them into the shape `(batch, seq_len, n_heads, headsize)`, then perform the attention calculation on each head, before arranging the final output back into shape `(batch, seq_len, n_heads * headsize)`.

```python
def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int):
    '''
    Implements multihead masked attention on the matrices Q, K and V.

    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)

    returns: shape (batch, seq, nheads*headsize)
    '''
    pass
```
""")

    with st.expander("Help - I'd like some more guidance for how to implement this function."):
        st.markdown("""Try and go through the following steps:

* Reshape `Q`, `K` and `V` into 4D tensors, as described
* Calculate the attention scores by multplying `Q` and `K`
* Apply the attention mask
* Apply softmax to get the attention probabilities
* Multiply by matrix `V` to get the attention values, then rearrange them back to a 3D tensor
""")

    st.markdown("""
---

## Attention block

Now, you should use this function in a `MultiheadMaskedAttention` block.

Note - it is a common practice to stack the matrices `W_Q`, `W_K`, `W_V` into a single linear layer, apply them all to the input tensor `x` at once, and then split `x` back into `Q`, `K`, `V` afterwards. In other words, rather than doing this:

""")

    st.image("ch1/images/computation_split.png", width=350)

    st.markdown("You can do this:")

    st.image("ch1/images/computation_parallel.png", width=580)

    st.markdown("""
You should use this method in your attention block implementation.

```python
class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''
        pass
```
""")

def section_2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#gpus">GPUs</a></li>
    <li><a class="contents-el" href="#building-custom-datasets">Building custom Datasets</a></li>
    <li><a class="contents-el" href="#choosing-config-parameters">Choosing <code>config</code> parameters</a></li>
    <li><a class="contents-el" href="#loss-functions">Loss functions</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""

# Putting it all together""")

    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown("""
Finally, you'll combine all the previous pieces into a full transformer architecture!

### Using `dataclasses`

`dataclasses` is a useful library in Python which allows you to create objects designed primarily for storing state, rather than for storing functions or logic. You can create a dataclass as follows:

```python
from dataclasses import dataclass

@dataclass
class InventoryItem:
    '''Class for keeping track of an item in inventory.'''
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
```

which will, when run, create a class with an `__init__()` that looks like:

```python
def __init__(self, name: str, unit_price: float, quantity_on_hand: int = 0):
    self.name = name
    self.unit_price = unit_price
    self.quantity_on_hand = quantity_on_hand
```

This can be useful for transformers, when trying to keep track of several different variables to do with the transformer architecture, such as number of heads, number of layers, size of the embedding dimension, etc. We've provided you with an object to manage this:

```python
@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05
```
""")

    st.markdown(r"""
A few notes:

* `head_size` is not in this config, because in our implementation we're assuming `num_heads * head_size = hidden_size`.
* `hidden_size` is also referred to as `embedding_dim`, or $d_\text{model}$ in some material you might have read.
* `max_seq_len` is used just to determine the size of the positional encoding matrix, when we create it in the `__init__` step of the `PositionalEncoding` module. When we actually add the positional encoding to our tensor `x`, we will take the first `seq_len` rows of it so that it's the same shape as `x`, rather than taking all `max_seq_len` rows.

Note that `seq_len` isn't referred to explicitly in any of the transformer architecture. The input sequence into our model can be of variable sequence length, and exactly how to interpret the output will depend on how you train the model. For instance, in the toy task in part 3, we will train the model to reverse a sequence: the input will be sequences of digits like `(0, 3, 1, 7, 9, 4)`, and the model's loss will be calculated by comparing its output to the reversed sequence `(4, 9, 7, 1, 3, 0)`. Later, when we train the model on the Shakespeare corpus, we will be interpreting the output in a different way: given a sequence of tokens `(token_1, ..., token_n)`, the model will be predicting `(token_2, ..., token_n+1)`. This will be discussed more later.
""")

    st.markdown("""
### Reviewing Positional Encoding

Now that you have this `config` object, you will be able to use its attributes in the subsequent layers you build. You should have already filled in this `PositionalEncoding` module from [yesterday's exercises](https://arena-w1d2.streamlitapp.com/Positional_Encoding):

```python
class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim: int, max_seq_len: int = 5000):
        pass

    def forward(self, x: Tensor) -> Tensor:
        '''
        x: shape (batch, seq_len, embedding_dim)
        '''
        pass
```

### DecoderBlock and DecoderOnlyTransformer

Now, you should implement `DecoderBlock` and `DecoderOnlyTransformer`. These are provided below for you, as diagrams.""")

    st.write("""<figure style="max-width:700px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNqNU21rwjAQ_ishnzuHZZ9kCA7dELoXUD-lY0Rz1WB6KWky5tb996WNrVbYWGmby3NP7u65JF90owXQEd0aXuzIcpoi8U_p1gFIqX9TDGj7dN4p1KvNM6rD0nAsM21yMH3yUu8BydXVmFQSLWzBVA00y9cgWGPers31uJkLidvXhjwRIjC8EZCp0YV2trHvwNg7pTf7kh1rCDMfaTyMSByRwWAQEXT5m-IHMGUIkaFi9xK5IkmNkidfb_A4ZCuEOmGYV-96w9dvpfyE6tnZwtm-qhddSiv1hTCPhqJbN1d_aTuFBBTnTe4a3OnsZ59jcdaIBahsjgiGTawFrPPWfThJJMMu76mdj8kL898FM-6Y8Yk6sRh6wMLw-ls1fkXf5YEu3v_U-pL6ITzQJGDNPxSUSARuhrVd3XyQnRQCMGzWwyxZhU25cIQ1cav9eJrYcex68pvQpubOoBH1Bz3nUvir81U7Ump3kENKR94UkHGnbH1zvj3VFYJbmAlptaGjjKsSIsqd1YsDbujIGgctaSq570V-ZH3_AN7pIx0" /></figure>""", unsafe_allow_html=True)

# graph TD
#     subgraph " "

#             subgraph DecoderOnlyTransformer
#             Token --> |integer|TokenEmbed[Token<br/>Embedding] --> AddEmbed[Add] --> Dropout --> BertBlocks[DecoderBlocks<br>1, 2, ..., num_layers] --> fnl[Final Layer Norm] --> un[Unembed] --> |vocab_size|Output
#             Position --> |integer|PosEmbed[Positional<br/>Embedding] --> AddEmbed
#         end

#         subgraph DecoderBlock
#             Input --> BertSelfInner[Attention<br>Layer Norm 1] --> Add[Add] --> MLdP[MLP<br>Layer Norm 2] --> Add2[Add] --> AtnOutput[Output]
#             Input --> Add
#             Add --> Add2
#         end

#         subgraph MLP
#             MLPInput[Input] --> Linear1 -->|4x hidden_size|GELU --> |4x hidden_size|Linear2 --> MLPDropout[Dropout] --> MLPOutput[Output]
#         end
#     end



    st.markdown("""
This very closely resembles the structure of GPT-2 (in fact, give or take a couple more dropout layers in your attention block, swap around the orders of the LayerNorm in the attention block, and with the right config parameters, you will actually have created GPT-2!).

A quick note on the order of the LayerNorms inside `DecoderBlock`. This diagram has followed the order in [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/#:~:text=generate%20it%3A-,The%20Residuals,-One%20detail%20in), but GPT-2 has them the other way around (i.e. LayerNorm before attention, and befor the MLP). You can read [this](https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab) for a more in-depth explanation of which one works better (tl;dr - they both work pretty well!).""")

    with st.expander("What should the 'normalized_shape' input of your LayerNorms be?"):
        st.markdown("""
You should have `normalized_shape = hidden_size`. From [Language Modelling with Transformers](https://docs.google.com/document/d/1j3EqnPnlg2g2z8fjst4arbZ_hLg_MgE0yFwdSoI237I/edit#heading=h.icw2n6iwy2of):

*Layer norm is going to take a vector of length embedding_dimension and scale it so that vector's components have a mean of 0 and a standard deviation close to 1. It's going to do this independently at each sequence position and independently for each batch element.*
""")

    st.markdown("""

Now, you should implement these blocks below. You can define a separate block for the MLP (this stands for [Multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)), or you can define all the components of the MLP within your `DecoderBlock` - whichever you prefer!

Once you've finished, you can progress to part 3 in order to test your model on a toy example.

```python
class DecoderBlock(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        pass
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        pass


class DecoderOnlyTransformer(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        pass
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        pass
```
""")

    with st.expander("Help - I'm confused about the tied unembeddings at the very end."):
        st.markdown("""
Your model should have **tied unembeddings**. This means that the matrix used at the start for token embedding should be the transpose of the matrix used at the end for unembedding (i.e. mapping the embedding dimension back to the vocabulary dimension), rather than being a separate matrix which has to be learned.

The easiest way to accomplish this is with an `einsum` using the weights of the token embedding matrix, which you should have defined in your `DecoderOnlyTransformer` with a line such as:

```python
self.token_embedding = nn.Embedding(...)
```

The weights can then be accessed using `self.token_embedding.weight`.
""")

def section_3():
    st.markdown("""

# Testing your model

<br>

From Jacob Hilton's curriculum:
""", unsafe_allow_html=True)

    st.info("""To check you have the attention mask set up correctly, train your model on a toy task, such as reversing a random sequence of tokens. The model should be able to predict the second sequence, but not the first.""")

    st.markdown("""

In other words, you can have an input sequence `t.tensor([1, 2, 4, 0, 3, 9])` with a target `t.tensor([9, 3, 0, 4, 2, 1])`, and you should expect your network to accurately predict the last three digits but not the first three.""")

    with st.expander("Why will your transformer predict the last but not first three?"):
        st.markdown("""Because this is a decoder-only transformer, which used masked attention. Each token is only allowed to read information from previous tokens. This means the 4th, 5th and 6th digits can read the 3rd, 2nd and 1st respectively to figure out what they should be, but the model has no way of figuring out what the first three digits should be.""")

    st.markdown("""
Again, we're not going to give you too much direction during this task, relative to the exercises of last week. You've already worked with datasets and dataloaders in last week's exercises, so this shouldn't be too much harder. 

### GPUs

You might find that this model trains faster if you use a GPU. You might have already used one if you did the resnet finetuning tutorial from last week. The standard way to define a `device` object in PyTorch is with code like this:

```python
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
```

You should now get `"cuda:0"` when you print your device. If you still get `cpu`, then you should examine your installation of PyTorch. The easiest way to install the right version of PyTorch is directly from the [PyTorch website](https://pytorch.org/). They give you a useful grid, and by selecting the right boxes you can get a command which you can run to install PyTorch with full GPU faculties:
""")

    st.image("ch1/images/install_pytorch.png")

    st.markdown("""

For instance, the selection above tells me that I should run the code:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

from my Anaconda Prompt, while in my working environment.

---

Once you've got your device working, this will introduce an entirely new way of getting bugs in your code: having two tensors (or a tensor and a model) not on the same device! The `to` method is useful here; it moves your object from one device to another (e.g. you can use `x.to(device)` to return a copy of `x` moved onto `device`). This same method can be used on models as well, i.e. `model.to(device)`. 

A word of warning here - when used on a model, this can be used as an inplace method or to return a new object (i.e. `model = model.to(device)`), but when used on tensors it **always returns a new tensor**, i.e. it isn't in place. This is why you might have seen lines like `y = y.to(device)` in training loops. 

There are also plenty of other ways to specify the device of a tensor, e.g. by using the `device` keyword argument when creating it, like in the following example:

```python
>>> t.ones(5, device=device)
tensor([1., 1., 1., 1., 1.], device='cuda:0')
```

We will go deeper into all these topics in the week on training at scale.

### Building custom Datasets

One note to help you get started - you can create your own dataset which inherits from PyTorch's `Dataset` object. For instance:

```python
from torch.utils.data import Dataset

class CustomTextDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        sample = (text, label)
        return sample
```

An explanation of each of these lines:

`class CustomTextDataset(Dataset)`

> Create a class called `CustomTextDataset`, which inherits from PyTorch's `Dataset` class.

`def __init__(self, text, labels)`

> When you initialise the class you need to import two variables. In this case, the variables are called `"text"` and `"labels"` to match the data which will be added.

```self.labels = labels & self.text = text```

> The imported variables can now be used in functions within the class by using `self.text` or `self.labels`.

`def __len__(self)`

> This function just returns the length of the labels when called. E.g., if you had a dataset with 5 labels, then the integer 5 would be returned.

`def __getitem__(self, idx)`

> This function is used by Pytorch's `Dataset` module to get a sample and construct the dataset. When initialised, it will loop through this function creating a sample from each instance in the dataset.

Once you've defined a dataset, you can define a DataLoader using this dataset just like you did last week. When you iterate through this object using `for item in dataloader`, the `item` will be the batched version of the output of `__getitem__`. In the case of the dataset above, the output is a tuple of `(text, label)`, so iterating through it using `for (text_batch, label_batch) in dataloader` will give you the batched versions of these two objects.

If you're interested, you can read much more [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) (although fair warning, this page contains much more than you're likely to need to know!).
""")

    with st.expander("Help - I'm not sure how I could use this to create a dataset for random sequences and their reverse."):
        st.markdown("""
Rather than being initialised with lists `text` and `labels`, you could initialise with variables `seq_len` and `total_size` (the latter of which would be your dataset length).

Your `__getitem__` function would return two items, `input` and `target`, which would respectively be a randomly chosen array of digits of length `seq_len`, and its reverse, respectively.
""")

    with st.expander("""Help - I'm getting the error "RuntimeError: CUDA error: device-side assert triggered"."""):
        st.markdown("""
This is a frustratingly ambiguous error, because it doesn't give you useful debugging information. Given the architecture you're working with, in this case it's most likely to be an error in your token or positional embedding, but more generally it indicates one of two possible errors:

* Inconsistency in the number of classes
* Wrong input for the loss function

To locate the exact line of the error (which by default this error message won't tell you), try restarting your kernel, and adding the following lines **before** you import PyTorch:

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
```
""")

    with st.expander("""Help - I'm getting the error "RuntimeError: CUDA out of memory"."""):
        st.markdown("""
First, try reducing the batch size (and restarting your kernel). If this doesn't work, you might want to switch to using the [GPU provided by Google Colab](https://www.tutorialspoint.com/google_colab/google_colab_using_free_gpu.htm) for the remainder of this exercise.
""")

    st.markdown("""
### Choosing `config` parameters

Some of the `config` parameters will be chosen for you, for example `vocab_size` will be equal to the number of possible digits which can be in your inputs. However, some of the other variables (e.g. `num_layers`, `num_heads`, and `hidden_size`).

It's likely that more than one possible set of values will successfully generate accurate predictions. My advice would be to look at the values used in the [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) paper, and start by scaling them down much smaller and seeing if this works.
""")

    with st.expander("Help - I'm still not sure what parameters to choose!"):
        st.markdown("""You shouldn't need more than 2 layers in your transformer (for more on why transformers with two layers are surprisingly powerful, wait until the interpretability section when we discuss [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)).
    
As for the `hidden_size`, experiment with values around 5-10x smaller than the size used in GPT-2.

This is obviously quite an unprincipled way to test different parameters, and over the next couple of days we'll look at more systematic methods of choosing hyperparameters using [Weights and Biases](https://wandb.ai/site).""")

    st.markdown("""
### Loss functions

Since you're predicting digits in a sequence, using cross entropy loss is most natural here. [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) returns a loss function which can accept inputs of the form `(logits, labels)`, where:

* `logits.shape = (batch, vocab_size)` (so `logits[i, :]` is a vector of `vocab_size` logits representing a distribution over possible labels)
* `labels.shape = (batch,)` (so the `i`th element of `labels` is the correct value)

If you've configured your transformer correctly, then from input of shape `(batch, seq_len)` you should get output `(batch, seq_len, 10)`. In other words, rather than computing a single probability distribution as your output, you're computing `seq_len` probability distributions (one for each digit in your sequence). The easiest way to deal with this is to flatten it into a tensor of shape `(batch * seq_len, 10)`, then this is a 2D array where the `[i, :]`th element is a vector of logits representing a distribution over one particular digit. You can then use this in your loss function. Similarly, the labels of the reversed sequences will be of shape `(batch, seq_len)`, and should be flattened to `(batch * seq_len,)`.
""")

func_list = [section_home, section_1, section_2, section_3]

page_list = ["🏠 Home", "1️⃣ Multihead Attention Block", "2️⃣ Putting it all together", "3️⃣ Testing your model"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()
