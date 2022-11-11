import streamlit as st
import platform
is_local = (platform.processor() != "")
rootdir = "" if is_local else "ch1/"

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
# Further investigations

Hopefully, last week you were able to successfully implement a transformer last week (or get pretty close!). If you haven't done that yet, then this should be your first priority going forwards with this week. **If you are struggling with getting your transformer to work, please send me (Callum) a link to your GitHub repo and I will be able to help troubleshoot.**

The rest of this week will involve continuing to iterate on your transformer architecture, as well as doing some more experiments with transformers. In the following pages, we'll provide a few suggested exercises. These range from highly open-ended (with no testing functions or code template provided) to highly structured (in the style of last week's exercises). 

All of the material here is optional, so you can feel free to do whichever exercises you want - or just go back over the transformers material that we've covered so far. **You should only implement them once you've done last week's tasks (in particular, building a transformer and training it on the Shakespeare corpus). 

Below, you can find a description of each of the set of exercises on offer. You can do them in any order, as long as you make sure to do exercises 1 and 2 at some point. Note that you can do e.g. 3B before 3A, but this is not advised since you'd have to import the solution from 3A and work with it, possibly without fully understanding the architecture.

---

### 1. Build and sample from GPT-2

As was mentioned in yesterday's exercises, you've already built something that was very close to GPT-2. In this task, you'll be required to implement an exact copy of GPT-2, and load in the weights just like you did last week for ResNet. Just like last week, this might get quite fiddly!

We will also extend last week's work by looking at some more advanced sampling methods, such as **beam search**.

### 2. Use your own modules

In week 0 we built a ResNet using only our own modules, which all inherited from `nn.Module`. Here, you have a chance to take this further, and build an entire transformer using only your own modules! This starts by having you define a few new blocks (`LayerNorm`, `Embedding` and `Dropout`), and then put them all together into your own transformer, which you can train on the Shakespeare corpus as in the task from last week.

### 3. Build and finetune BERT

BERT is an encoder-only transformer, which has a different kind of architecture (and a different purpose) than GPT-2. In this task, you'll build a copy of BERT and load in weights, then train it on 

Once you've built BERT, you'll be able to train it to perform well on tasks like classification and sentiment analysis.

""")

def section1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#gpt-architecture-an-overview">GPT architecture: an overview</a></li>
    <li><a class="contents-el" href="#notes-on-copying-weights">Copying weights</a></li>
    <li><a class="contents-el" href="#testing-gpt">Testing GPT</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""

## Introduction

Here, you have an opportunity to rewrite your transformer implementations so that they exactly match the architecture of GPT-2, then load in GPT's weights and biases to your own model just like you did in the first week with ResNet. You can re-use some of your ResNet code, e.g. the function to copy over weights (although you will have to rewrite some of it; more on this below).

We will be using the GPT implementation from [HuggingFace](https://huggingface.co/docs/transformers/model_doc/auto). They provide a repository of pretrained models (often, transformer models) as well as other valuable documentation.

```python
transformers.AutoModelForCausalLM.from_pretrained("gpt2")
```

## GPT architecture: an overview

First, we will start by going through the differences between GPT and your implementation (i.e. the diagram from [W1D3](https://arena-w1d3.streamlitapp.com/Putting_it_all_together#decoderblock-and-decoderonlytransformer)).

* The order of the LayerNorms in the decoder block have changed: they now come *before* the attention and MLP blocks, rather than after.
* The attention block has two dropout layers: one immediately after the softmax (i.e. before multiplying by `V`), and one immediately after multiplying with `W_O` at the very end of the attention block. Note that the dropout layers won't actually affect weight-loading or performance in eval mode (and you should still be able to train your model without them), but all the same it's nice to be able to exactly match GPT's architecture!
* All your linear layers should have biases - even though some of them are called projections (which would seem to suggest not having a bias), this is often how they are implemented.
* GPT-2 uses a learned positional embedding (i.e. `nn.Embedding`) rather than a sinusoidal one.""")

    with st.expander("Question - how do you think you would use a positional encoding during a forward pass, if it was an 'nn.Embedding' object?"):
        st.markdown("""
When we used a sinusoidal encoding, we simply took a slice of the first `seq_len` rows. When using `nn.Embedding`, the equivalent is to pass in `t.arange(seq_len)`. So our first step in `forward` should look something like:

```python
pos = t.arange(x.shape[1], device=x.device)
x = self.token_embedding(x) + self.positional_encoding(pos)
```
""")

    st.markdown("""
We've provided you with the function `utils.print_param_count(*models)`, which can be passed a list of models (ideally two: yours and the pretrained GPT), and displays a color-coded dataframe making it easy to see which parameters match and which don't:

```python
my_gpt = GPT(config).train()
gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").train()

utils.print_param_count(my_gpt, gpt)
```

Ideally, this will produce output that looks something like this (up to possibly having different layer names):
""")

    # st.image("ch1/images/gpt-compared.png")
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        st.image("ch1/images/gpt-compared.png")

    st.markdown("""

Note - the `utils.print_param_count` function works slightly differently than it did in the ResNet utils file, `w0d3/utils`. This is because it iterates through `model.parameters()` rather than `model.state_dict()`. For why it does this, read the next section.

## Copying weights

Here is the template for a function which you should use to copy over weights from pretrained GPT to your implementation. 

```python
def copy_weights_from_gpt(my_gpt: GPT, gpt) -> GPT:
    '''
    Copy over the weights from gpt to your implementation of gpt.

    gpt should be imported using: 
        gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

    Returns your gpt model, with weights loaded in.

    You might find the function `copy_weights` from w0d3 helpful as a template.
    '''
    
    # FILL IN CODE: define a state dict from my_gpt.named_parameters() and gpt.named_parameters()

    my_gpt.load_state_dict(state_dict)
    return my_gpt
```

A few notes here, regarding how this function will be different from the copying weights function you were given in w0d3:

* The linear layer weights are actually transposed between GPT and your implementation (you can see this from row 4 of the table above). This applies to the linear layers with square weight matrices too, so take care to transpose when copying these weights over! (Note that the embeddings aren't transposed; only the linear layers.)
* It's easier to iterate through using `model.named_parameters()` rather than `model.state_dict()` in this case. 
    * Optional exercise: inspect `gpt.state_dict()` and `gpt.named_parameters()`, and see what objects are in one but not the other. Why are these objects in one but not the other, and why do you think it's better to use `named_parameters` than `state_dict` when copying over? (or you can skip this exercise, and just take my word for it!)
""")

    with st.expander("Answer to optional exercise"):
        st.markdown("""
Upon inspection, we find:

```python
state_dict_names = set(gpt.state_dict().keys())
param_names = set(dict(gpt.named_parameters()).keys())

print(len(state_dict_names))            # 173
print(len(param_names))                 # 148
print(param_names < state_dict_names)   # True
```

So `state_dict` is a superset of `parameters`, and it contains a lot more objects. What are these objects? When we print out the elements of the set difference, we see that they are all biases or masked biases of the transformer layer. If we inspect the corresponding objects in `gpt.state_dict().values()`, we see that these are nothing more than the objects we've been using as attention masks (i.e. a triangular array of 1s and 0s, with 1s in the positions which aren't masked). We clearly don't need to copy these over into our model!

The moral of the story - `state_dict` sometimes contains things which we don't need, and it depends heavily on the exact implementation details of the model's architecture.

Note that we're in the fortunate position of not having any batch norm layers in this architecture, because if we did then we couldn't get away with using `parameters` (they are buffers, not parameters).
""")

    st.markdown("""
## Testing GPT

Once you've copied over your weights, you can test your GPT implementation.

GPT2's tokenizer uses a special kind of encoding scheme called [byte-level byte-pair encoding](https://docs.google.com/document/d/1XJQT8PJYzvL0CLacctWcT0T5NfL7dwlCiIqRtdTcIqA/edit#heading=h.dgmfzuyi6796). You should import a tokenizer, and test your model, using the following code:

```python
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
utils.test_load_pretrained_weights(my_gpt, tokenizer)
```

The testing function above gives your GPT the prompt `"Former President of the United States of America, George"`, and tests whether the next predicted tokens contain `" Washington"` and `" Bush"` (as they are expected to). You are encouraged to look at this function in `utils`, and understand how it works.

---

If you get this working, you can try fine-tuning your GPT on text like your Shakespeare corpus. See if you can use this to produce better results than your original train-from-scratch, tokenize-by-words model.
""")

def section2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#imports">Imports</a></li>
    <li><a class="contents-el" href="#modules">Modules</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#nn-embedding">nn.Embedding</a></li>
        <li><a class="contents-el" href="#nn-gelu">nn.GELU</a></li>
        <li><a class="contents-el" href="#nn-layernorm">nn.LayerNorm</a></li>
        <li><a class="contents-el" href="#nn-dropout">nn.Dropout</a></li>
    </li></ul>
    <li><a class="contents-el" href="#putting-it-all-together">Putting it all together</a></li>
    <li><a class="contents-el" href="#bonus">Bonus</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""

## Imports

```python
import torch as t
from torch import nn, optim
import plotly.express as px
from typing import Optional, Union, List
import utils
```

## Modules

Here, we'll complete the process of defining our own modules, which all inherit from `nn.Module`. By the end of this, we'll have built enough modules to be able to construct our entire transformer.

The modules that we still haven't defined, which we'll need for our decoder-only transformer, are:
```
nn.Embedding
nn.GELU
nn.LayerNorm
nn.Dropout
```

We'll go through these one by one.

### `nn.Embedding`

Implement your version of PyTorch's [`nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) module. The PyTorch version has some extra options in the constructor, but you don't need to worry about these.

```python
class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        pass

    def forward(self, x: t.LongTensor) -> t.Tensor:
        '''For each integer in the input, return that row of the embedding.
        '''
        pass

    def extra_repr(self) -> str:
        pass

utils.test_embedding(Embedding)
```

### `nn.GELU`

Now, you should implement GELU. This should be very similar to your implementation of ReLU in the first week. You can use either of the two approximations recommended [here](https://paperswithcode.com/method/gelu). You can use `utils.plot_gelu` to verify your function works as expected.

```python
class GELU(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        pass

utils.plot_gelu(GELU)
```

### `nn.LayerNorm`

Use the [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for Layer Normalization to implement your own version which exactly mimics the official API. Use the biased estimator for `var` as shown in the docs.

```python
class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-05, elementwise_affine: bool = True):
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        pass

utils.test_layernorm_mean_1d(LayerNorm)
utils.test_layernorm_mean_2d(LayerNorm)
utils.test_layernorm_std(LayerNorm)
utils.test_layernorm_exact(LayerNorm)
utils.test_layernorm_backward(LayerNorm)
```

### `nn.Dropout`

Finally, implement Dropout in accordance with its [documentation page](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html). You will have to implement it differently depending on whether you are in training mode, just like you did for BatchNorm. You can check whether you're in training mode with `self.training`.

```python
class Dropout(nn.Module):

    def __init__(self, p: float):
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        pass

utils.test_dropout_eval(Dropout)
utils.test_dropout_training(Dropout)
```

## Putting it all together

Now, you're ready to go back over your decoder-only transformer, and replace all the `nn.Module`s with your own! Try to train your model on the reversed digits task, and then on the Shakespeare corpus.

## Bonus

As a bonus task, you can try and implement GPT using only your own modules! Unfortunately, this isn't as simple as taking your code from the GPT implementation task (assuming you've already done this) and swapping out the `nn.Module` calls. GPT's architecture is more complicated than the ResNet that we've been working with, and will involve a lot more fussing around with messy details to get exactly right!

Later this week, I'll post a longer set of instructions on how you might go about doing this. However, if you're feeling adventurous, then you might want to give this task ago before I post this! The following code might help:

```python
import transformers
import pandas as pd

gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

with pd.option_context("display.max_rows", None):
    display(pd.DataFrame([
        {"name": name, "shape": param.shape, "param count": param.numel()}
        for name, param in gpt.named_parameters()
    ]))
```

Note that `torchinfo` unfortunately seems to break on gpt2 (if anyone finds a way to get it working, please ping me a message!).

""")

def section3():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#bert-architecture-an-overview">BERT architecture: an overview</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#mlm-and-special-tokens">MLM and special tokens</a></li>
    </li></ul>
    <li><a class="contents-el" href="#other-architectural-differences-for-bert">Other architectural differences for BERT</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#attention-masking-for-pad">Attention masking for [PAD]</a></li>
        <li><a class="contents-el" href="#tokentype-embedding">TokenType embedding</a></li>
        <li><a class="contents-el" href="#unembedding">Unembedding</a></li>
    </li></ul>
    <li><a class="contents-el" href="#bert-config">BERT config</a></li>
    <li><a class="contents-el" href="#copying-weights">Copying weights</a></li>
    <li><a class="contents-el" href="#testing-bert">Testing BERT</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""

## Introduction

So far, we've only looked at decoder-only transformers (your own implementations, and GPT-2). Today, we'll take a look at encoder-only transformers.

BERT (Bidirectional Encoder Representations from Transformers) is the most famous in a line of Muppet-themed language research, originating with [ELMo](https://arxiv.org/pdf/1802.05365v2.pdf) (Embeddings from Language Models) and continuing with a series of increasingly strained acronyms:

* [Big BIRD](https://arxiv.org/pdf/1910.13034.pdf) - Big Bidirectional Insertion Representations for Documents
* [Ernie](https://arxiv.org/pdf/1904.09223.pdf) - Enhanced Representation through kNowledge IntEgration
* [Grover](https://arxiv.org/pdf/1905.12616.pdf) - Generating aRticles by Only Viewing mEtadata Records
* [Kermit](https://arxiv.org/pdf/1906.01604.pdf) - Kontextuell Encoder Representations Made by Insertion Transformations

Today you'll implement your own BERT model such that it can load the weights from a full size pretrained BERT, and use it to predict some masked tokens.

You can import (and inspect) BERT using:

```python
bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
```

Note, there might be some error warnings that appear when you import `bert`, don't worry about these.

## BERT architecture: an overview

The diagram below shows an overview of BERT's architecture. This differs from the [W1D3](https://arena-w1d3.streamlitapp.com/Putting_it_all_together#decoderblock-and-decoderonlytransformer) architecture in quite a few ways, which we will discuss below.
""")

    st.write("""<figure style="max-width:1000px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNqNVNlq4zAU_RWhZzclYZ7CEGgmaQm400LTJ3kocnTjiFiSkaWhadN_rxbbqRPPUGPQ1d10ztHyjjeKAZ7iQtNqh9aLTCL31TaPjgy7P5PR24vMQZuUysLSAu5dj_KU47-VrKy5JWH4g66uZqHglxJCyTk52TF2BJEDY1wWqOZvcMxJyiVQ_TPXs7tl-uzHlB5Ao99KCz9bc2DoWYa6psdftaF5rH8gqSq4QQ_W-PVP0ECyr2wuGEVQ_YS12oOMS3BpoAB9DK6lX5sE0yG6ni1bDhHPDWMxwxl9_DG-0KpS1nTazEu12dfkZPqicYImCRqNRgmSVryUvkc9rFnk2of-qGpuuDpD77wRWRum5f8IDKixPlQwoIh3u5KoCfKz7_U925XejgQtBs5Wp9sTlNuVlKDJjTEgPZ9unUvt0TgG79PHNmkykDVpWhgZZSUXJ6kPxHXo2n2PlwPQb-Ycod_XOxNvwdjbxx-vaMcZc8KG7fbXIm7BWSDWTFqWzTEjzdix_xevgLkxcIIFaEE5cy_Eu3dn2OxAQIanzmSwpbY0_oH4cKm2YtTAknGjNJ5uaVlDgqk16ukgN3hqtIU2acGpU0I0WR-fvPhwKw" /></figure>""", unsafe_allow_html=True)

    # graph TD
    #     subgraph " "

    #         subgraph BertLanguageModel
    #             InputF[Input] --> BertCommonB[BertCommon] --> |embedding size|b[Linear<br>GELU<br>Layer Norm<br>Tied Unembed] --> |vocab size|O[Logit Output]
    #         end

    #             subgraph BertCommon
    #             Token --> |integer|TokenEmbed[Token<br/>Embedding] --> AddEmbed[Add<br>Layer Norm] --> Dropout --> BertBlocks[BertBlocks<br>1, 2, ..., num_layers] --> |embedding size|Output
    #             Position --> |integer|PosEmbed[Positional<br/>Embedding] --> AddEmbed
    #             TokenType --> |integer|TokenTypeEmb[Token Type<br/>Embedding] --> AddEmbed
    #         end

    #         subgraph BertBlock
    #             Input --> BertSelfInner[Attention] --> Add[Add<br>Layer Norm 1] --> MLP --> Add2[Add<br>Layer Norm 2] --> AtnOutput[Output]
    #             Input --> Add --> Add2
    #         end

    #         subgraph BertMLP
    #             MLPInput[Input] --> Linear1 -->|4x hidden size|GELU --> |4x hidden size|Linear2 --> MLPDropout[Dropout] --> MLPOutput[Output]
    #         end
    #     end

    st.markdown("""
Conceptually, the most important way in which BERT and GPT differ is that BERT has bidirectional rather than unidirectional masking in its attention blocks. This is related to the tasks BERT is pretrained on. Rather than being trained to predict the next token in a sequence like GPT, BERT is pretrained on a combination of **masked language modelling** and **next sentence prediction**, neither of which require forward masking.

Note - this diagram specifically shows the version of BERT which performs masked language modelling. Many different versions of BERT share similar architectures. In fact, this language model and the classifier model which we'll work with in subsequent exercises both share the section we've called `BertCommon` - they only differ in the few layers which come at the very end.

""")

    st.write("""<figure style="max-width:500px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNqVUktvwjAM_itRznDpsZo4AGOaVOikwandwW1MsdQkVR6TGPDfl6RMiAObZkWy4_dnfSfeaoE8552B4cC2y1qxINY3o6Pm4dVq9N5F5mhcAarz0OE69OhvOVEEGWwdacW28_vIqxq8W1VJfbDpdJZaLbSUWs2rmz3GzigbFIJUxyx94bmpClII5qkxs5fnYhd1AUc0bKONjL8toWA7lequPT51C81YX1aF7six0rs4_7YaKvEQ56IHa2lPaP4HMnuAMvsbZlatyFjH3rSlNKFU_THCWxo9aO8S7nSJawPlJWvjnmjPZVb9rNzCWP0L3GDyCZdoJJAIVDhFd83dASXWPA-mwD343kUmXEIqeKffj6rluTMeJ9wPAhwuCcLFJM_30NvgRUFOm_VIr8SyyzcIHsrZ" /></figure>""", unsafe_allow_html=True)

    # graph TD
    # subgraph " "

    #     subgraph BertLanguageModel
    #         direction TB
    #         InputF[Input] --> BertCommonB[BertCommon] --> |embedding size|b[Linear<br>GELU<br>Layer Norm<br>Tied Unembed] --> |vocab size|O[Logit Output]
    #     end

    #     subgraph BertClassifier
    #         direction TB
    #         InputF2[Input] --> BertCommonB2[BertCommon] --> |embedding size|b2[First Position Only<br>Dropout<br>Linear] --> |num classes|O2[Classification Output]
    #     end

    # end


    st.markdown("""
### MLM and special tokens

You can import BERT's tokenizer using the following code:

```python
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
```

You should read [this page](https://albertauyeung.github.io/2020/06/19/bert-tokenization.html/) to get an idea of how the tokenizer works, then answer the following questions to check your understanding. You might also find [this paper](https://arxiv.org/abs/1810.04805) on BERT helpful.

""")

    with st.expander("Run 'tokenizer.tokenize' on some string. What do you notice about the output, whenever it has length greater than 1?"):
        st.markdown("""
When the length is greater than one, all tokens other than the first are prefixed with `##`. This is to indicate that the token is a suffix following some other subwords, rather than being the start of a word.
""")

    st.markdown("""If you run `tokenizer.special_tokens_map`, you will produce the following output:

```
{'unk_token': '[UNK]',
 'sep_token': '[SEP]',
 'pad_token': '[PAD]',
 'cls_token': '[CLS]',
 'mask_token': '[MASK]'}
```

Question - what do each of these tokens mean?""")

    with st.expander("[UNK]"):
        st.markdown("""`[UNK]` means unknown token. You can run something like `tokenizer.tokenize("🤗")` to produce this token.""")

    with st.expander("[CLS]"):
        st.markdown("""`[CLS]` stands for classification. Recall that the output of a transformer is a set of logits representing distributions over the vocabulary - one logit vector for each token in the input. The logit vector corresponding to `[CLS]` is the one we use to calculate predictive loss during fine-tuning on classification tasks (as well as pre-training on next-sentence prediction). """)

    with st.expander("[SEP]"):
        st.markdown("""`[SEP]` (separator) is appended to the end of sentences during NSP, in order to indicate where one sentence finishes and the next starts. It is also used in classification (here it is appended to the end of the sentence), but this is less important.""")

    with st.expander("[PAD]"):
        st.markdown("""`[PAD]` tokens are appended to the end of a sequence to bring its length up to the maximum allowed sequence length. This is important because we need all data to be the same length when batching them.""")

    with st.expander("[MASK]"):
        st.markdown("""`[MASK]` is used for masked language modelling. For instance, if our data contained the sentence `"The lemon was yellow and tasted sour"`, we might convert this into `"The [MASK] was yellow and tasted sour"`, and then measure the cross entropy between the model's predicted token distribution at `[MASK]` and the true answer `"lemon"`.""")

    st.markdown("""
A few more specific questions about tokenization, before we move on to talk about BERT architecture. Note that some of these are quite difficult and subtle, so don't spend too much time thinking about them before you reveal the answer.

In appendix **A.1** of [the BERT paper](https://arxiv.org/abs/1810.04805), the masking procedure is described. 15% of the time the tokens are masked (meaning we process these tokens in some way, and perform gradient descent on the model's predictive loss on this token). However, masking doesn't always mean we replace the token with `[MASK]`. Of these 15% of cases, 80% of the time the word is replaced with `[MASK]`, 10% of the time it is replaced with a random word, and 10% of the time it is kept unchanged. This is sometimes referred to as the 80-10-10 rule.
""")

    with st.expander("What might go wrong if you used 100-0-0 (i.e. only ever replaced the token with [MASK]) ?"):
        st.markdown("""
In this case, the model isn't incentivised to learn meaningful encodings of any tokens other than the masked ones. It might focus only on the `[MASK]` token and ignore all the others. 

As an extreme example, your model could hypothetically reach zero loss during pretraining by  perfectly predicting the masked tokens, and just **outputting a vector of zero logits for all non-masked tokens**. This means you haven't taught the model to do anything useful. When you try and fine-tune it on a task which doesn't involve `[MASK]` tokens, your model will only output zeros!

In practice, this extreme case is unlikely to happen, because for meaningful computation to be done at the later transformer layers, the residual stream for the non-`[MASK]` tokens must also contain meaningful information. See the comment below these dropdowns.
""")

    with st.expander("What might go wrong if you used 80-0-20 (i.e. only [MASK] and unchanged tokens) ?"):
        st.markdown("""
In the above case, the model isn't incentivised to store any information about non-masked tokens. In this case, the model **isn't incentivised to do anything other than copy those tokens**. This is because it knows that any non-masked token is already correct, so it can "cheat" by copying it. This is bad because the model's output for non-masked tokens won't capture anything about its relationship with other tokens.

The "extreme failure mode" analogous to the one in the 100-0-0 case is that the model learns to perfectly predict the masked tokens, and copy every other token. This model is also useless, because it hasn't learned any contextual relationships between different tokens; it's only learned to copy.
""")

    with st.expander("What might go wrong if you used 80-20-0 (i.e. only [MASK] and random tokens) ?"):
        st.markdown("""
In this case, the model will treat `[MASK]` tokens and the random tokens it has to predict as exactly the same. This is because it has no training incentive to treat them differently; both tokens will be different to the true token, but carry no information about the true token.

So the model would just treat all tokens the same as `[MASK]`, and we get the same extreme failure mode as in the 100-0-0 case.
""")

    st.markdown("""
It's worth noting that all of these points are pretty speculative. The authors used trial and error to arrive at this masking method and these proportions, and there is [some evidence](https://arxiv.org/pdf/2202.08005.pdf) that other methods / proportions work better. Basically, we still don't understand exactly why and how masking works, and to a certain extent we have to accept that!""")


    st.markdown("""
## Other architectural differences for BERT

Here are another few differences, on top of the bidirectional attention.

### Attention masking for `[PAD]`

Although we don't apply forward masking, we do still need to apply a form of masking in our attention blocks.

Above, we discussed the `[PAD]` token, which is used to extend sequences so that they have the maximum length. We don't want to read and write any information from/to the padding tokens, and so we should apply a mask to set these attention scores to a very large negative number before taking softmax. Here is some code to implement this:

```python
class MultiheadAttention(nn.Module):

    def __init__(self, config: TransformerConfig):
        pass

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor]) -> t.Tensor:
        pass 


class BERTBlock(nn.Module):

    def __init__(self, config):
        pass

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, nheads=1, seqQ=1, seqK)
        '''
        pass


def make_additive_attention_mask(one_zero_attention_mask: t.Tensor, big_negative_number: float = -10000) -> t.Tensor:
    '''
    one_zero_attention_mask: 
        shape (batch, seq)
        Contains 1 if this is a valid token and 0 if it is a padding token.

    big_negative_number:
        Any negative number large enough in magnitude that exp(big_negative_number) is 0.0 for the floating point precision used.

    Out: 
        shape (batch, nheads=1, seqQ=1, seqK)
        Contains 0 if attention is allowed, big_negative_number if not.
    '''
    pass

utils.test_make_additive_attention_mask(make_additive_attention_mask)
```

An explanation of this code: `additive_attention_mask` is the tensor which gets added to the attention scores before taking softmax (unless it is `None`, in which case we don't do any masking). It has the purpose of zeroing all the attention probabilities corresponding to padding tokens. You should implement the function `make_additive_attention_mask`, which creates `additive_attention_mask` from your input data.

A few notes on the implementation of `make_additive_attention_mask`:

* You should make a mask to zero the attention probabilities at position `(q_idx, k_idx)` if and only if `k_idx` corresponds to a padding token.
    * Mathematically, this means that each row of your attention probabilities will only have non-zero values in the positions corresponding to non-`[PAD]` tokens.
    * Conceptually, this means that your model won't be paying attention to any of the padding tokens.
* Your `additive_attention_mask` can be of shape `(batch, 1, 1, seqK)`, because it will get broadcasted when added to the attention scores. 
""")

    # with st.expander("Question - what would be the problem with adding the attention mask at position (q_idx, k_idx) if EITHER q_idx OR k_idx corresponded to a padding token, rather than only adding at if k_idx is a padding token?"):
    #     st.markdown("""
    # If you did this, then in the softmax stage you'd be taking softmax over a row of all `-t.inf` values (or extremely small values). This would produce `nan` outputs (or highly numerically unstable outputs).
    # """)

    st.markdown("""
Also, a note on using `nn.Sequential`. If this is how you define your BertBlocks, then you might run into a problem when you try and call `self.bertblocks(x, additive_attention_mask)`. This is because `nn.Sequential` can only take one input which is sequentially fed into all its blocks. The easiest solution is to manually iterate over all the blocks in `nn.Sequential`, like this:

```python
for block in self.bertblocks:
    x = block(x, additive_attention_mask)
```

You can also use a `nn.ModuleList` rather than a `nn.Sequential`. You can think of this as an `nn.Sequential` minus the ability to run the entire thing on a single input (and plus the existence of an `append` method, which Sequential doesn't have). You can also think of `nn.ModuleList` as a Python list, but with the extra ability to register its contents as modules. For instance, using `self.layers = [layer1, layer2, ...]` won't work because the list contents won't be registered as modules and so won't appear in `model.parameters()` or `model.state_dict()`. But if you use `self.layers = nn.ModuleList([layer1, layer2, ...])`, you don't have this problem.

### TokenType embedding

Rather than just having token and positional embeddings, we actually take a sum of three embeddings: token, positional, and **TokenType embedding**. This embedding takes the value 0 or 1, and has the same output size as the other embeddings. It's only used in **Next Sentence Prediction** to indicate that a sequence position belongs to the first or second sentence, respectively. Here is what your `BertCommon` should look like:

```python
class BertCommon(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        pass
            
    def forward(
        self,
        x: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        '''
        input_ids: (batch, seq) - the token ids
        one_zero_attention_mask: (batch, seq) - only used in training, passed to `make_additive_attention_mask` and used in the attention blocks.
        token_type_ids: (batch, seq) - only used for NSP, passed to token type embedding.
        '''
        pass
```

You can see how this is applied in the diagram in the [BERT paper](https://arxiv.org/abs/1810.04805), at the top of page 5. This also illustrates how the `[CLS]` and `[SEP]` tokens are used.
""")

    with st.expander("Question - what size should the TokenType embedding matrix have?"):
        st.markdown("""It should be initialised as `nn.Embedding(num_embeddings=2, embedding_dim=hidden_size)`, because the input will be an array of indices with value either zero or one.
    
    See below for the value of `hidden_size`, and the other architectural parameters.""")

    st.markdown("""

### Unembedding

Rather than simply ending with `LayerNorm -> Tied Unembed`, the Bert Language Model ends with a sequence of `Linear -> GELU -> LayerNorm -> Tied Unembed`. Additionally, the tied unembedding in BERT has a bias (which isn't tied to anything, i.e. it's just another learned parameter). The best way to handle this is to define the bias as an `nn.Parameter` object with size `(vocab_size,)`. Unfortunately, this seems to make copying weights a bit messier. I think `nn.Parameter` objects are registered first even if they're defined last, so you might find the output of `utils.print_param_count(my_bert, bert)` is shifted by 1 (see image below for what my output looked like), and you'll need to slightly rewrite the function you used to copy weights from GPT (more on this below).""")

    cols = st.columns([1, 10, 1])
    with cols[1]:
        st.image("ch1/images/bert-compared.png")

    # with st.expander("""Question - how does HuggingFace's BERT model implement the tied encoding? You will have to inspect their BERT model in order to find the answer."""):
    #     st.markdown("""
    # If you print out BERT, then at the start you can see:

    # ```
    # BertForMaskedLM(
    #   (bert): BertModel(
    #     (embeddings): BertEmbeddings(
    #       (word_embeddings): Embedding(28996, 768, padding_idx=0)
    #       ...
    # ```

    # and at the end you can see:

    # ```
    #   ...
    #   (cls): BertOnlyMLMHead(
    #     (predictions): BertLMPredictionHead(
    #       ...
    #       (decoder): Linear(in_features=768, out_features=28996, bias=True)
    #     )
    #   )
    # )
    # ```

    # If you compare the weights of these layers (using `bert.bert.embeddings.word_embeddings` and `bert.cls.predictions.decoder`), you can see that the weights are exactly the same:

    # ```python
    # unembed_weight = bert.cls.predictions.decoder.weight
    # embed_weight = bert.bert.embeddings.word_embeddings.weight

    # t.testing.assert_close(unembed_weight, embed_weight.T)
    # ```

    # We can conclude that this BERT model implements the unembedding as a linear layer with bias.


    # """)

    st.markdown(r"""
## BERT config

Referring to the [BERT paper](https://arxiv.org/abs/1810.04805), and try and find the values of all the `TransformerConfig` parameters. Note that we are using $\text{BERT}_\text{BASE}$ rather than $\text{BERT}_\text{LARGE}$. The `layer_norm_epsilon` isn't mentioned in the paper, but can be found by examining the BERT model. Also, the `vocab_size` figure given in the paper is just approximate - you should inspect your tokenizer to find the actual vocab size.
""")

    with st.expander("Answer"):
        st.markdown("""

* `num_layers`, `num_heads` and `vocab_size` can be found at the end of page 3
* `max_seq_len` and `dropout` are mentioned on page 13
* `vocab_size` can be found via `tokenizer.vocab_size`
* `layer_norm_epsilon` can be found by just printing `bert` and inspecting the output

We find that our parameters are:

```python
config = TransformerConfig(
    num_layers = 12,
    num_heads = 12,
    vocab_size = 28996,
    hidden_size = 768,
    max_seq_len = 512,
    dropout = 0.1,
    layer_norm_epsilon = 1e-12
)
```
""")

    st.markdown(r"""
## Copying weights

Again, the process of copying over weights from BERT to your model is a bit messy! A quick summary of some of the points made above, as well as a few more notes:

* It's easier to iterate through using `model.named_parameters()` rather than `model.state_dict()` in this case. 
    * Optional exercise: inspect `bert.state_dict()` and `bert.named_parameters()`, and see what objects are in one but not the other. Why are these objects in one but not the other, and based on this can you see why it's better to use `named_parameters` than `state_dict` when copying over? (or you can skip this exercise, and just take my word for it!)
""")

    with st.expander("Answer to optional exercise"):
        st.markdown("""
Upon inspection, we find:

```python
state_dict_names = set(bert.state_dict().keys())
param_names = set(dict(bert.named_parameters()).keys())

print(len(state_dict_names))  # 205
print(len(param_names))       # 202

print(state_dict_names - param_names)
# {'bert.embeddings.position_ids', 'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias'}
```

From this output, we conclude that `state_dict` is a superset of `parameters`, and contains three items not in `parameters`.

Inspecting `bert.embeddings.position_ids`, we see that it is just an unsqueezed `t.arange(512)` array, which we can guess gets passed into the positional encoding.

Inspecting the other two, we see that they belong to the linear layer at the very end of BERT, i.e. the unembedding with bias. `cls.predictions.decoder.weight` is a duplicate of the token embedding matrix at the start (but this object is only counted once in `bert.parameters()`, because it's a tied embedding). `cls.predictions.decoder.bias` is a duplicate of `cls.predictions.bias`, and we don't need to count this twice! Again, there is only one underlying parameter.

The moral of the story - not only can `state_dict` sometimes contain things we don't need, but it can also sometimes contain multiple objects which refer to the same underlying parameters.
""")

    st.markdown("""
* Unlike for GPT, you shouldn't have to transpose any of your weights.
* BERT's attention layer weights are stored as `W_Q`, `W_K`, `W_V`, `W_O` separately (in that order) rather than as `W_QKV`, `W_O`. You should do the same in your model, because it makes iterating over and copying weights easier.

Here is a code template which you can fill in to write a weight copying function. Referring to `copy_weights` from W0D3 might be helpful.

```python
def copy_weights_from_bert(my_bert: BertLanguageModel, bert: transformers.models.bert.modeling_bert.BertForMaskedLM) -> BertLanguageModel:
    '''
    Copy over the weights from bert to your implementation of bert.

    bert should be imported using: 
        bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")

    Returns your bert model, with weights loaded in.
    '''
    
    # FILL IN CODE: define a state dict from my_bert.named_parameters() and bert.named_parameters()

    my_bert.load_state_dict(state_dict)
    return my_bert
```

## Testing BERT

Once you've built your BERT architecture and copied over weights, you can test it by looking at its prediction on a masked sentence. Below is a `predict` function which you should implement. You might find it useful to inspect the `utils.test_load_pretrained_weights` function (which is used for the GPT exercises) - although if you can get by without looking at this function, you should try and do so.

Note that you might have to be careful with your model's output if you want `predict` to work on your BERT and the imported BERT. This is because (just like GPT) the imported BERT will output an object which has a `logits` attribute, rather than just a tensor of logits.""")

    st.markdown(r"""
```python
def predict(model, tokenizer, text: str, k=15) -> List[List[str]]:
    '''
    Return a list of k strings for each [MASK] in the input.
    '''
    pass

def test_bert_prediction(predict, model, tokenizer):
    '''Your Bert should know some names of American presidents.'''
    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text)
    print(f"Prompt: {text}")
    print("Model predicted: \n", "\n".join(map(str, predictions)))
    assert "Washington" in predictions[0]
    assert "Bush" in predictions[0]

test_bert_prediction(predict, my_bert, tokenizer)
```
""")

func_list = [section_home, section1, section2, section3]

page_list = ["🏠 Home", "1️⃣ Build and sample from GPT-2", "2️⃣ Use your own modules", "3️⃣ Build and finetune BERT"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()
