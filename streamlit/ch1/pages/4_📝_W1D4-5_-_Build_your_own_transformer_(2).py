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
# Building your own transformer (2)

Hopefully, yesterday you were able to successfully implement a transformer (or get pretty close!). Now, you'll train your transformer on a more difficult task: **autoregressive text generation** on the [complete works of William Shakespeare](https://www.gutenberg.org/files/100/100-0.txt).

This is the task recommended by Jacob Hilton in his curriculum.

In today's material, we'll also take you through ways to generate output from a transformer (i.e. methods for taking a set of logits representing a distribution over tokens, and returning a single token). This includes methods like greedy sampling, and top-k and top-p sampling.

## Autoregressive text generation

Your transformer's output will have the same `seq_dim` as the input. This is actually helpful, because it means you can use your transformer as follows: given input `(token_1, token_2, ..., token_n)`, you can predict `(token_2, token_3, ..., token_n+1)`.

This means that your model is actually learning `n` different tasks: how to predict the `j+1`th token from the first `j`, for each of `j = 1, 2, ..., n`. The masked attention makes sure the model can't "cheat" by looking at future tokens. If you've set up your positional encoding correctly, then this should work even when the number of tokens you input into your model is not always the same.

When you're generating output autoregressively, you can input a sequence of tokens with dimension `(1, seq_len)` (since our batch size is 1). Your output will be a set of logits of size `(1, seq_len, vocab_size)`. You should sample based on the `[0, -1, :]`th element of these logits, because this corresponds to the distribution over tokens your model predicts for the `(seq_len+1)`th token.

An example: suppose your initial prompt was `"The apple was"`. Sampling in this way (continually using the `[0, -1, :]`th logit vector to generate your next output) might look like this:

```
"The apple was" -> "red"
"The apple was red" -> "and"
"The apple was red and" -> "tasted"
"The apple was red and tasted" -> "sweet"
```

How do you use a logit vector (i.e. a distribution over possible tokens) to sample a single token? This is the subject of the first section, on sampling. In the second section, we'll combine everything we've done so far, and train our model on the Shakespeare corpus.""")

def section1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#sampling-boilerplate">Sampling Boilerplate</a></li>
    <li><a class="contents-el" href="#greedy-search">Greedy Search</a></li>
    <li><a class="contents-el" href="#sampling-with-categorical">Sampling with <code>Categorical</code></a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#temperature">Temperature</a></li>
        <li><a class="contents-el" href="#frequency-penalty">Frequency Penality</a></li>
        <li><a class="contents-el" href="#sampling-manual-testing">Sampling - Manual Testing</a></li>
    </ul></li>
    <li><a class="contents-el" href="#top-k-sampling">Top-K Sampling</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#top-k-sampling-example">Top-K Sampling - Example</a></li>
    </ul></li>
    <li><a class="contents-el" href="#top-p-aka-nucleus-sampling">Top-p aka Nucleus Sampling</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#top-p-sampling-example">Top-p Sampling - Example</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Sampling

One obvious method to sample tokens from a distribution would be to always take the token assigned the highest probability. But this can lead to some boring and repetitive outcomes, and at worst it can lock our transformer's output into a loop.

First, you should read HuggingFace's blog post [How to generate text: using different decoding methods for language generation with Transformers
](https://huggingface.co/blog/how-to-generate). Once you've done that, we've included some exercises below that will allow you to write your own methods for sampling from a transformer. Some of the exercises are strongly recommended (two asterisks), some are weakly recommended (one asterisk) and others are perfectly fine to skip if you don't find these exercises as interesting.

We will be working with the [HuggingFace implementation](https://huggingface.co/docs/transformers/index) of classic transformer models like GPT. You might have to install the transformers library before running the cells below. You may also want to go back and complete exercise 2 in the [Tokenisation and Embedding exercises](https://arena-ldn-w1d1.streamlitapp.com/Tokenisation_and_embedding) from W1D1.

```python
import torch as t
import torch.nn.functional as F
import transformers

gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
```

## Sampling Boilerplate

The provided functions `apply_sampling_methods` and `sample_tokens` include the boilerplate for sampling from the model. Note that there is a special token `tokenizer.eos_token`, which during training was added to the end of a each article. GPT-2 will generate this token when it feels like the continuation is at a reasonable stopping point, which is our cue to stop generation.

The functions called in `apply_sampling_methods` are not defined yet - you are going to implement them below.

```python

def apply_sampling_methods(
    input_ids: t.Tensor, logits: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.
x
    input_ids: shape (seq,)
    '''
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)

def sample_tokens(
    model,
    tokenizer,
    initial_text: str,
    max_tokens_generated: int = 30,
    **kwargs
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.int64, device=device)
        new_input_ids_truncated = new_input_ids[-min(tokenizer.model_max_length, new_input_ids.shape[0]):].unsqueeze(0)
        output = model(new_input_ids_truncated)
        all_logits = output if isinstance(output, t.Tensor) else output.logits
        logits = all_logits[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        assert isinstance(new_token, int)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)
```

A few notes on this function:

* We use `tokenizer.encode` to convert the initial text string into a list of logits. You can also pass the argument `return_tensors="pt"` in order to return the output as a tensor.
* `new_input_ids` is a concatenation of the original input ids, and the ones that have been autoregressively generated.
* `new_input_ids_truncated` truncates `new_input_ids` at `max_seq_len` (because you might get an error at the positional embedding stage if your input sequence length is too large).
* The line `all_logits = ...` is necessary because HuggingFace's GPT doesn't just output logits, it outputs an object which contains `logits` and `past_key_values`. In contrast, your model will probably just output logits, so we can directly define logits as the model's output.
""")

    with st.expander("Question - why do we take logits[0, -1] ?"):
        st.markdown("""
Our model input has shape `(batch, seq_len)`, and each element is a token id. Our output has dimension `(batch, seq_len, vocab_size)`, where the `[i, j, :]`th element is a vector of logits representing a prediction for the `j+1`th token.

In this case, our batch dimension is 1, and we want to predict the token that follows after all the tokens in the sequence, hence we want to take `logits[0, -1, :]`.
""")

    st.markdown("""
### Greedy Search

Implement `greedy_search`, which just returns the most likely next token. If multiple tokens are equally likely, break the tie by returning the smallest token.

Why not break ties randomly? It's nice that greedy search is deterministic, and also nice to not have special code for a case that rarely occurs (floats are rarely exactly equal).

Tip: the type checker doesn't know the return type of `item()` is int, but you can assert that it really is an int and this will make the type checker happy.

```python
def greedy_search(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    pass

prompt = "Jingle bells, jingle bells, jingle all the way"
print("Greedy decoding with prompt: ", prompt)
output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
print(f"Your model said: {output}")
expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
assert output == expected

print("Greedy decoding a second time (should be deterministic): ")
output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
print(f"Your model said: {output}")
expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
assert output == expected

print("Tests passed!")
```

## Sampling with `Categorical`

PyTorch provides a [`distributions` package](https://pytorch.org/docs/stable/distributions.html#distribution) with a number of convenient methods for sampling from various distributions.

For now, we just need [`t.distributions.categorical.Categorical`](https://pytorch.org/docs/stable/distributions.html#categorical). Use this to implement `sample_basic`, which just samples from the provided logits (which may have already been modified by the temperature and frequency penalties).

Note that this will be slow since we aren't batching the samples, but don't worry about speed for now.

```python
def sample_basic(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    pass

N = 20000
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_basic(unnormalized_logits) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, probs, atol=0.01, rtol=0)
print("Tests passed!")
```

### Temperature

Temperature sounds fancy, but it's literally just dividing the logits by the temperature.

```python
def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    pass

logits = t.tensor([1, 2]).log()
cold_logits = apply_temperature(logits, 0.001)
print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
t.testing.assert_close(cold_logits, 1000.0 * logits)
hot_logits = apply_temperature(logits, 1000.0)
print("A high temperature flattens the distribution: ", hot_logits)
t.testing.assert_close(hot_logits, 0.001 * logits)
print("Tests passed!")
```
""")

    with st.expander("Question - what is the limit of applying 'sample_basic' after adjusting with temperature, when temperature goes to zero? How about when temperature goes to infinity?"):
        st.markdown("""
The limit when temperature goes to zero is greedy search (because dividing by a small number makes the logits very big, in other words the difference between the maximum logit one and all the others will grow). 

The limit when temperature goes to infinity is uniform random sampling over all words (because all logits will be pushed towards zero).")
""")

    st.markdown(r"""
### Frequency Penalty

The frequency penalty is simple as well: count the number of occurrences of each token, then subtract `freq_penalty` for each occurrence. Hint: use `t.bincount` (documentation [here](https://pytorch.org/docs/stable/generated/torch.bincount.html)) to do this in a vectorized way.""")

    with st.expander("""Help - I'm getting a RuntimeError; my tensor sizes don't match."""):
        st.markdown("""
Look at the documentation page for `t.bincount`. You might need to use the `minlength` argument - why?""")

    st.markdown(r"""
```python
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    pass

bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
input_ids = tokenizer.encode(bieber_prompt, return_tensors="pt").squeeze()
logits = t.ones(tokenizer.vocab_size)
penalized_logits = apply_freq_penalty(input_ids, logits, 2.0)
assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"
print("Tests passed!")
```

### Sampling - Manual Testing

Run the below cell to get a sense for the `temperature` and `freq_penalty` arguments. Play with your own prompt and try other values.

Note: your model can generate newlines or non-printing characters, so calling `print` on generated text sometimes looks awkward on screen. You can call `repr` on the string before printing to have the string escaped nicely.

```python
N_RUNS = 1
your_prompt = "Jingle bells, jingle bells, jingle all the way"
cases = [
    ("High freq penalty", dict(freq_penalty=100.0)),
    ("Negative freq penalty", dict(freq_penalty=-1.0)),
    ("Too hot!", dict(temperature=2.0)),
    ("Pleasantly cool", dict(temperature=0.7)),
    ("Pleasantly warm", dict(temperature=0.9)),
    ("Too cold!", dict(temperature=0.01)),
]
for (name, kwargs) in cases:
    for i in range(N_RUNS):
        output = sample_tokens(gpt, tokenizer, your_prompt, max_tokens_generated=24, **kwargs)
        print(f"Sample {i} with: {name} ({kwargs}):")
        print(f"Your model said: {repr(output)}\n")
```

## Top-K Sampling

Conceptually, the steps in top-k sampling are:
- Find the `top_k` largest probabilities
- Set all other probabilities to zero
- Normalize and sample

Your implementation should stay in log-space throughout (don't exponentiate to obtain probabilities). This means you don't actually need to worry about normalizing, because `Categorical` accepts unnormalised logits.
""")

    with st.expander("Help - I don't know what function I should use for finding the top k."):
        st.markdown("Use [`t.topk`](https://pytorch.org/docs/stable/generated/torch.topk.html).")

    st.markdown("""
```python
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    pass

k = 3
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_top_k(unnormalized_logits, k) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
expected = probs.clone()
expected[:-k] = 0
expected /= expected.sum()
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, expected, atol=0.01, rtol=0)
print("Tests passed!")
```

### Top-K Sampling - Example
The [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) famously included an example prompt about unicorns. Now it's your turn to see just how cherry picked this example was.

The paper claims they used `top_k=40` and best of 10 samples.

```python
your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
output = sample_tokens(gpt, tokenizer, your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)
print(f"Your model said: {repr(output)}")
```

## Top-p aka Nucleus Sampling

Conceptually, in top-p sampling we:

- Sort the probabilities from largest to smallest
- Find the cutoff point where the cumulative probability first equals or exceeds `top_p`. We do the cutoff inclusively, keeping the first probability above the threshold.
- If the number of kept probabilities is less than `min_tokens_to_keep`, keep that many tokens instead.
- Set all other probabilities to zero
- Normalize and sample

Optionally, refer to the paper [The Curious Case of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751.pdf) for some comparison of different methods.


""")

    with st.expander("Help - I'm confused about how nucleus sampling works."):
        st.markdown("""The basic idea is that we choose the most likely words, up until the total probability of words we've chosen crosses some threshold. Then we sample from those chosen words based on their logits.
    
For instance, if our probabilities were `(0.4, 0.3, 0.2, 0.1)` and our cutoff was `top_p=0.8`, then we'd sample from the first three elements (because their total probability is `0.9` which is over the threshold, but the first two only have a total prob of `0.7` which is under the threshold). Once we've chosen to sample from those three, we would renormalise them by dividing by their sum (so the probabilities we use when sampling are `(4/9, 3/9, 2/9)`.""")

    with st.expander("Help - I'm stuck on how to implement this function."):
        st.markdown("""First, sort the logits using the `sort(descending=True)` method (this returns values and indices). Then you can get `cumulative_probs` by applying softmax to these logits and taking the cumsum. Then, you can decide how many probabilities to keep by using the `t.searchsorted` function.
    
Once you've decided which probabilities to keep, it's easiest to sample from them using the original logits (you should have preserved the indices when you called `logits.sort`). This way, you don't need to worry about renormalising like you would if you were using probabiliities.""")

    st.markdown("""
```python
def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    pass

N = 2000
unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
samples = t.tensor([sample_top_p(unnormalized_logits, 0.5) for _ in range(N)])
counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
print("top_p of 0.5 or lower should only return token 2: ", counts)
assert counts[0] == 0 and counts[1] == 0

N = 2000
unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
samples = t.tensor([sample_top_p(unnormalized_logits, 0.50001) for _ in range(N)])
counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
print("top_p in (0.5, 0.8] should return tokens 1 and 2: ", counts)
assert counts[0] == 0

N = 4000
top_p = 0.71
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_top_p(unnormalized_logits, top_p) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
expected = probs.clone()
expected[0:2] = 0
expected /= expected.sum()
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, expected, atol=0.01, rtol=0.0)

print("All tests passed!")
```

### Top-p Sampling - Example

```python
your_prompt = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
output = sample_tokens(gpt, tokenizer, your_prompt, temperature=0.7, top_p=0.95, max_tokens_generated=64)
print(f"Your model said: {repr(output)}")
```
""")

def section2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#defining-a-dataset">Defining a dataset</a></li>
    <li><a class="contents-el" href="#defining-a-tokenizer">Defining a tokenizer</a></li>
    <li><a class="contents-el" href="#final-notes">Final notes</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Training your transformer on Shakespeare

Now that we've discussed sampling, we can proceed to train our transformer on the Shakespeare corpus!

## Defining a dataset

You can access the complete works of Shakespeare at [this link](https://www.gutenberg.org/files/100/100-0.txt). You should tokenize the corpus as recommended by Jacob, using `re.split(r"\b", ...)`. If you're unfamiliar with how to use Python regular expressions, you might want to read [this w3schools tutorial](https://www.w3schools.com/python/python_regex.asp)

When creating a dataset (and dataloader) from the Shakespeare corpus, remember to tokenize by splitting at `"\b"`. You can follow most of the same steps as you did for your reversed digits dataset, although your dataset here will be a bit more complicated.
""")

    with st.expander("Help - I'm not sure how to convert my words into token ids."):
        st.markdown("""Once you tokenizer the corpus, you can define a list of all the unique tokens in the dataset. You can then sort them alphabetically, and tokenize by associating each token with its position in this sorted list.""")

    st.markdown("""
Training on the entire corpus might take a few hours. I found that you can get decent results (see below) from just training on 1% of the corpus, which only takes a few minutes. Your mileage may vary.

## Defining a tokenizer

If you want to use the functions from the previous section to sample tokens from your model, you'll need to construct a tokenizer from your dataset. As a minumum, your tokenizer should have methods `encode` and `decode`, and a `model_max_length` property. Here is a template you can use:

```python
class WordsTokenizer():
    model_max_length: int

    def __init__(self, wordsdataset: WordsDataset):
        pass

    def encode(self, initial_text: str, return_tensors: Optional[str] = None) -> Union[list, t.Tensor]:
        '''
        Tokenizes initial_text, then returns the token ids.
        
        Return type is list by default, but if return_tensors="pt" then it is returned as a tensor.
        '''
        pass

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        '''
        Converts ids to a list of tokens, then joins them into a single string.
        '''
        pass
```

Note that if you want to call your tokenizer like `tokenizer(input_text)`, you'll need to define the special function `__call__`.

## Final notes

This task is meant to be open-ended and challenging, and can go wrong in one of many different ways! If you're using VSCode, this is probably a good time to get familiar with the [debugger](https://code.visualstudio.com/docs/editor/debugging) if you haven't already! It can help you avoid many a ... winter of discontent.

When choosing config parameters, try starting with the same ones described in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Next week, we'll look at [wandb](https://wandb.ai/site), and how it gives you a more systematic way of choosing parameters.

Once you succeed, you'll be able to create riveting output like this (produced after about 10 mins of training):

```python
initial_text = "turn down for what"

# Defining my transformer
model = DecoderOnlyTransformer(config).to(device).train()
# Defining my own tokenizer as function of trainset (see bullet point above)
tokenizer = WordsTokenizer(trainset)

text_output = sample_tokens(model, tokenizer, initial_text, max_tokens_generated=100, temperature=1.0, top_k=10)

print(text_output)

# turn down for what you do you think,
# That take the last, of many, which is so much I
# As this blows along than my life thou say’st, which makes thy hand,
# Thou wilt be given, or more
# Entitled in thy great world’s fresh blood will,
# To answer th’ alluring countenance, beauty 
```

Also, if you manage to generate text on the Shakespeare corpus, you might want to try other sources. When I described this task to my sister, she asked me what would happen if I trained the transformer on the set of all Taylor Swift songs - I haven't had time to try this, but bonus points if anyone gets good output by doing this.
""")

func_list = [section_home, section1, section2]

page_list = ["🏠 Home", "1️⃣ Sampling", "2️⃣ Training your transformer on Shakespeare"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()
