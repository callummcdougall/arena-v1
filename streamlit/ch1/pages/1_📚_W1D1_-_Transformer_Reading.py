import streamlit as st

import numpy as np
import plotly.express as px
import platform
is_local = (platform.processor() != "")
rootdir = "" if is_local else "ch1/"

st.set_page_config(layout="wide")

if "pe" not in st.session_state:
    pe = np.load(rootdir + "images/pe.npy")
    pe_sim = np.einsum("ab,cb->ac", pe, pe) / np.outer(np.linalg.norm(pe, axis=-1), np.linalg.norm(pe, axis=-1))
    st.session_state["pe"] = pe
    st.session_state["pe_sim"] = pe_sim
    st.session_state["fig_viz"] = px.imshow(pe, color_continuous_scale="RdBu")
    st.session_state["fig_dot_product"] = px.imshow(pe_sim, color_continuous_scale="Blues")
fig_viz = st.session_state["fig_viz"]
fig_dot_product = st.session_state["fig_dot_product"]

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
# Transformer Reading

Today is mainly focused on reading up on transformers. You'll be going through the reading material, and making sure you understand most of the core concepts you'll need to implement your own transformer as you progress through this week.

We strongly recommend you make notes on this material as you go through it. You'll probably have a lot of questions, and you might also find it helpful to post these questions in the Slack channel so that people can help answer them.

### Recommended reading

* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
    * You might prefer watching the video (although it misses out some details). 
* [Language Modelling with Transformers](https://docs.google.com/document/d/1XJQT8PJYzvL0CLacctWcT0T5NfL7dwlCiIqRtdTcIqA/edit#)
    * This google doc contains a lot of useful information about transformers, although some of the diagrams and schematics are less clear than in the Illustrated Transformer.

Don't worry if you don't follow all of this yet. A lot of it is bound to still be confusing, but should become less so as you work through your own implementation of a transformer.

Once you've read through these once or twice, try going through the questions in the pages on the sidebar. There are no mandatory coding exercises today (although a couple of optional ones); the focus should be on the reading.""")

    st.info("""**The questions have been labelled according to their priorities. \*\* indicates recommended material, \* indicates optional (but higher priority than the rest), and no asterisks indicates lower priority. You should try and spend most of your working time today reading the two documents above, rather than just going through these questions.**""")

    st.markdown("""

Lastly, here is a diagram explaining the attention mechanism. It should also be useful over the next few days, when you'll be writing functions to implement attention.

""")

    with st.expander("View diagram"):
        st.image("ch1/images/attention_diagram.png")
        st.markdown(r"""
1. The query is a feature vector that describes what we're looking for in the sequence, i.e. what we might want to pay attention to
2. The key is a feature vector describing what each element might be “offering”, or why it is important
3. We take inner product over key and query vectors, to get **attention scores** for each query-key pair.
4. Dividing by $\sqrt{h}$ prevents vanishing or exploding gradient problems.
5. We mask wherever `q < k` by setting the logits to a very small value, because these query-key pairs correspond to a token reading information from a future token, which is “cheating”
6. Because we applied softmax over the `s_k` dim, and now we're summing over `s_k`, this step can be seen as taking a [convex combination](https://en.wikipedia.org/wiki/Convex_combination) of value vectors. In practice, this is often just equivalent to selecting the value vector with the highest probability (if one attention score is much larger than the others), but we still need to make the function differentiable!
7. This is the step where we in some sense “aggregate over the heads”. You can write out this linear operation as doing `n` linear operations (one for each head, and each one using a different chunk of matrix `W_O`, then taking the sum of the results of these operations.
""")

    st.markdown("""
### Optional reading

If you've got time at the end of today, here's a few other things you might want to read. All of these provide different approaches to explaining transformers, and you might find some of them easier to understand than others.

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Section 3 of the paper that introduced the transformer explains the architecture well. Don't worry too much about the encoder and how that fits in, as that's somewhat specific to translation – unsupervised transformer language models are generally decoder-only.
* [Formal Algorithms for Transformers](https://arxiv.org/pdf/2207.09238.pdf) - A DeepMind paper covering what transformers are, how they are trained, what they are used for, their key architectural components, and an overview of the most prominent models.
* [The Transformer Family](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/) - An overview of many transformer variants, including Transformer-XL, Image Transformer, Sparse Transformer, Reformer and Universal Transformer.

Another option you have is to brainstorm more [muppet-related names](https://www.theverge.com/2019/12/11/20993407/ai-language-models-muppets-sesame-street-muppetware-elmo-bert-ernie) for transformers. Bonus points go to especially creative names.
""")
 
def section_1():
    st.markdown(r"""

# Transformers (general)

## 1. \*\*

What differentiates transformers from RNNs and LSTMs?""")

    with st.expander("""Answer"""):
        st.markdown("""RNNs are neural networks which contain cycles, allowing for the processing of temporal or sequential information.
    
LSTMs are a more sophisticated type of RNN which uses **gates** to better regulate the flow of information through neurons.

Both RNNs and LSTMs process information word-by-word, which makes them hard to parallelize when processing an entire sentence or passage. They also have trouble capturing certain kinds of **long and short-range dependencies**. 

Transformers on the other hand process every token in a sequence at once, via the **self-attention mechanism**. This enables the transformer to attend to the whole sequence, and thus use the context of the entire sequence during each intermediate step.

They also explicitly use a **positional encoding**.""")

# st.markdown("""
# ## 2.

# In Anthropic's transformer circuits paper, [they state that](https://transformer-circuits.pub/2021/framework/index.html#notation:~:text=Privileged%20Basis%20vs%20Basis%20Free) the only vectors in a transformer with a **privileged basis** are tokens, attention patterns, and MLP activations. Can you figure out why these three are privileged? (Hard!)
# """)

# with st.expander("Answer"):
#     st.markdown("""The MLP is privileged because we use activation functions (e.g. ReLU) on it. This applies an operation on individual neurons, which isn't invariant to rotation.

# While token embeddings can be freely rotated and so don't have a privileged basis, the tokens themselves correspond to bits of text in a sequence, with each possible value for a token representing a different string (e.g. words or characters). For instance, the basis vector corresponding to the token `103` in GPT2's vocabulary specifically means `[MASK]`, whereas the linear combination of basis vectors is not meaninfgul in the same way.

# The attention pattern has had a softmax function applied to it, making the sums along one of the dimensions equal to 1. This is not rotationally invariant.

# (Note - I'm not 100% sure about this answer, this is just my best guess)""")

    st.markdown(r"""## 2. \*

Does BERT use the encoder and decoder parts of the transformer, as described in **The Illustrated Transformer**? How about GPT?
""")

    with st.expander("Answer"):
        st.markdown("""
BERT only uses the encoder, not the decoder. This is because it's built for tasks like classification, rather than text generation.

GPT is decoder-only. It is more suitable for text-generation.
""")

    st.markdown(r"""
## 3. 

Why do we have so many skip connections, especially connecting the input of an attention function to the output? Intuitively, what if we didn't?
""")

    with st.expander("Answer"):
        st.markdown("""
Earlier in this course, while looking at [ResNets](https://arxiv.org/abs/1512.03385?context=cs), we discussed the **degradation problem** for neural networks, and how skip connections empirically seemed to fix it. One framing is that skip connections make it easier for a network to learn the identity function, and thereby preserve information from earlier layers.

If we didn't include these skip connections, we might experience a degradation of performance for very deep transformer models due to vanishing / exploding gradients.
""")
 
def section_2():
    st.markdown(r"""
# Training

## 1. \*\*

What is the difference between masked language modelling and next sentence prediction? 
""")

    with st.expander("Answer"):
        st.markdown("""These are two types of self-supervised pre-training tasks.

**Masked language modelling** (MLM) = part of the text is masked, and must be predicted. For instance, `the [MASK] was yellow and tasted sour` should give high probability to `lemon` being in the `[MASK]` position.

**Next sentence prediction** (NSP) = take two sentences which are either adjacent in the dataset or from two different sections, and predict which case they are in.
""")

    st.markdown(r"""
## 2. \*

How might you generate text from a masked language model? Check if your procedure is related to anything in [this paper](https://arxiv.org/pdf/1902.04094.pdf).
""")

    st.markdown(r"""
## 3. \*

What is the difference between **supervised** and **self-supervised** learning? Give an example of both, for NLP.
""")

    with st.expander("Answer"):
        st.markdown("""In self-supervised learning, we don't require human input to perform data-labelling. 

An example of supervised learning is sentiment analysis (or at least it is usually supervised). Most forms of text-classification are also supervised.

Examples of self-supervised learning are **masked language modelling** and **next sentence prediction** (see the answer to question 1).
""")
 
def section_3():
    st.markdown(r"""
# Attention mechanism

## 1. \**

In the formula for attention:
$$
\alpha=\operatorname{Attention}(Q, K, V)=\operatorname{SoftMax}\left(\frac{Q K^{\top}}{\sqrt{d_\text{head}}}\right) V
$$

why do we divide by $\sqrt{d_\text{head}}$?""")

    with st.expander("Answer"):
        st.markdown(r"""This leads to having more stable gradients. Dividing the matrix by $\sqrt{d_\text{head}}$ before taking softmax in essence means we're making the probabilities less extreme, and less delicately dependent on the input vectors $Q, K$.
    
We can get more mathematically precise. With a few simplifying assumptions, it's possible to show that the dot product of each query and key vector has a variance of around $d_\text{head}$, so dividing by $\sqrt{d_\text{head}}$ makes the variances approximately 1.""")

    st.markdown(r"""## 2. \**

In the formula for attention (see above), what are the dimensions for $Q$, $K$, and $V$? (you can refer to the diagram in `🏠 Home` if you're confused).
""")

    with st.expander("Answer"):
        st.markdown("""
    If batch size is 1, then $Q$ and $K$ have dimensions `(seq_len, d_k)`. `seq_len` is the number of tokens in each input sequence to the model, and `d_k` is the dimension of the query/key vectors (in the **Language Modelling with Transformers** document, this is referred to as **head_size**). $V$ has dimension `(seq_len, d_v)`, where `d_v` is the dimension of the vectors representing the values. This doesn't have to be the same as `d_k`, although in practice it often is (e.g. for GPT-2 and PaLM, as well as for the transformers which you will be building).

    If batch size is greater than 1, then all these matrices should additionally have the batch size as their 0th dimension.
    """)
 
def section_4():
    st.markdown(r"""
# Tokenisation and embedding

For some more intuition about tokenisation, and what the embedding dimension actually represents geometrically, you can read [this post](https://towardsdatascience.com/word2vec-explained-49c52b4ccb71) explaining **word2vec**, one of the first and most well-known historical examples of learned word embeddings. You don't need to worry about the probabilistic model used to generate these embeddings, just the general idea of how this space can preserve semantic meaning. You might also want to try out [Semantle](https://semantle.com/), which is a game similar to Wordle except that rather than returning the number and position of correct letters, it gives you the semantic similarity of your guess to the target word (based on the word2vec embedding).

The token embedding layer at the start of most transformer models is pretty conceptually similar to embeddings like word2vec. Both are learned via gradient descent on a self-supervised task, usually involving classification or prediction. There are a few differences though, for instance:

* Word2vec tokenizes by words, whereas most transformers tokenize into smaller units, e.g. [Wordpiece](https://paperswithcode.com/method/wordpiece) for BERT.
* Exactly how embeddings are learned depends on the architecture of the neural network in which they are applied. Word2vec is learned as part of a shallow, 2-layer neural network, much simpler than the transformer architectures which it predates.

However, since these two are pretty conceptually similar, [some models](https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76) will initialise their embedding layers by loading in weights from pre-learned embeddings, and then letting them be updated in accordance with the specific supervised task.
""")

    st.markdown(r"""
## 1. \*\*

What is the difference between embeddings like word2vec, and the **output** of an encoder-only transformer like BERT?

(Note, I'm talking about the output of a transformer, not the token embedding which makes up just a single layer of the transformer.)
""")

    with st.expander("Answer:"):
        st.markdown("""
Word2vec only ever ouputs the same vector for each word. 

In contrast, BERT's outputs are **context-dependent**. Tt will output a vector in the embedding dimension for each word in its input, but this vector will depend on the other words in the sequence.
""")

    st.markdown("""
## 2. \*\*

At the end of the transformer architecture, we perform an **unembedding** to take us from the embedding dimension back to vectors of logits over the set of tokens in our vocabulary. 

Sometimes we use a **tied unembedding**, which means we use the transpose of the original token embedding matrix rather than learning a new embedding matrix. What do you think is the main advantage of doing this?
""")

    with st.expander("Answer"):
        st.markdown("""The main advantage is saving space on your processor by not needing another matrix (and with it a completely different set of learned parameters.""")

        st.info("""Interestingly, another much less obvious advantage is that you can probe the network by applying the unembedding matrix at earlier points in the network's computation, which gives an idea of the result so far (intuitively, because the embedding s[ace] has the "same meaning" at the start and end of the network, so it is likely to have the same meaning in the middle as well) - this is called the [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).""")

    st.markdown(r"""
## 3. \*

Try playing around with tokenizers in Python, using the `transformers` library. The following code can get you started:

```python
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
```

(note, you might need to install the `transformers` library in your environment)

Try running `tokenizer.encode`, `tokenizer.decode`, `tokenizer.tokenize`, and just `tokenizer` on inputs. Can you figure out what each of these functions does?

## 4. \*

Look up PyTorch's [Embedding module](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html). Try playing around with it, and getting a feel for how it works.
""")

    with st.expander("Question - what is the difference between nn.Linear and nn.Embedding?"):
        st.markdown("""
`nn.Linear` is just a linear transformation, mapping `in_features` to `out_features`. It doesn't change the dimensionality of the input.

`nn.Embedding` treats each number in the input as an index, and returns a corresponding vector in the embedding dimension. So it will produce output with one more dimension than the input.

Crucially, the difference is **not** that `nn.Linear` is updated by gradient descent and `nn.Embedding` is static. Both layers' weights are updated by gradient descent in the same way; this is how the embedding matrix learns semantic representations for the tokens (from the initialisation values of the embdding vectors, which are randomly chosen).
""")

    st.markdown("""
## 5.

The [Language Modelling with Transformers](https://docs.google.com/document/d/1j3EqnPnlg2g2z8fjst4arbZ_hLg_MgE0yFwdSoI237I/edit#) document said that:

*"in high dimensional space you can have exponentially many vectors that are almost orthogonal"*

Make this statement more precise: how does the expected cosine similarity between two random vectors relate to the dimension $D$?
""")

    with st.expander("Hint"):
        st.markdown("""Start by considering the unit vector (1, 0, 0, …) and another random unit vector.""")

    with st.expander("Answer"):
        st.markdown(r"""The expected square of the cosine similarity between (1, 0, 0, …) and another random unit vector $v$ is $\frac{1}{D}$ (assuming the latter unit vector had each of its elements following the same distribution). This is relatively straightforward to prove: the formula for the square of cosine similarity in this case evaluates to $v_1^2$, and the expected value of this must be $\frac{1}{D}$ because when you sum over the squared elements you get 1 (since $v$ is normalised).

You can then argue that any reasonable way of choosing two random vectors has the same property, because you can choose a rotation to apply to both vectors which sends the first of your random vectors to (1, 0, 0, …); your second vector should still be a random unit vector, and rotations won't affect the cosine similarity.""")

def section_5():
    st.markdown(r"""
# Positional encoding

## 1.

**Sinusoidal positional encoding** is computed as follows:
$$
\mathrm{PE}(i, \delta)= \begin{cases}\sin \left(\frac{i}{10000^{2 \delta^{\prime} / d}}\right) & \text { if } \delta=2 \delta^{\prime} \,\text{ for some }\delta^{\prime} \\ \cos \left(\frac{i}{10000^{2 \delta^{\prime} / d}}\right) & \text { if } \delta=2 \delta^{\prime}+1 \,\text{ for some }\delta^{\prime}\end{cases}
$$
You can see a plot of it below, with $L$ (the token position in the sequence) on the y-axis and $d$ (the embedding dimension) on the x-axis:
""")

    st.plotly_chart(fig_viz, use_container_width=True)

    st.markdown("""
Each row of this matrix forms a vector in the embedding dimension corresponding to a particular token position in the sequence. This vector is added to each corresponding token's embedding before any of the transformer's linear layers or attention blocks.

Try to plot a graph of sinusoidal positional encoding, like the one above. Use the same parameters $L = 32$ and $d = 128$.""")


    with st.expander("Click to reveal code, if you're stuck."):
        st.markdown("""
```python
n = 10000
d = 128
L = 32

def PE(delta):
    
    sin_array = np.sin(delta / n ** (2 * np.arange(d//2) / d))
    cos_array = np.cos(delta / n ** (2 * np.arange(d//2) / d))
    
    array = np.zeros(d)
    array[::2] = sin_array
    array[1::2] = cos_array
    
    return array

array_2d = np.zeros((L, d))
for k in range(L):
    array_2d[k] = PE(k)
    
px.imshow(array_2d, color_continuous_scale="Gray")


# or more efficiently:

angles = np.outer(np.arange(L), 1 / n ** (2 * np.arange(d//2) / d))

array_2d = np.zeros((L, d))
array_2d[:, ::2] = np.sin(angles)
array_2d[:, 1::2] = np.cos(angles)

px.imshow(array_2d, color_continuous_scale="Gray")
```    
""")

    st.markdown("""
## 2.

One nice property of sinusoidal position encoding is that the distance between neighbouring timesteps are symmetrical and decays nicely with time. Here is a graph illustrating this, by showing the dot product:
""")

    st.plotly_chart(fig_dot_product, use_container_width=True)

    st.markdown("""
Can you reproduce this graph? Can you show mathematically why this should be the case?

You might find it helpful to refer back to the `w0d1` material on Fourier transforms.""")

    with st.expander("Code to reproduce graph, from previous code:"):
        st.markdown("""
```python
def cosine_similarity(vec1, vec2):
    
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_dot_product_graph(array_2d):

    (L, d) = array_2d.shape

    arr = np.zeros((L, L))

    # Note, there are also more elegant ways to do this than with a for loop!
    for i in range(L):
        for j in range(L):
            arr[i, j] = cosine_similarity(array_2d[i], array_2d[j])

    px.imshow(arr, color_continuous_scale="Blues").show()
```
""")
 
def section_6():
    st.markdown("""
# Layer Normalisation and Dropout

[This reading](https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm) provides a useful illustration of LayerNorm vs BatchNorm in NLP, and also explains some of the background as to why LayerNorm is used in NLP (and specifically transformers), whereas BatchNorm tends to be preferred for computer vision tasks.

## 1. \*\*

If we have language data with dimension `(batch, sequence, embedding)`, which dimensions does Layer Norm normalize over? How many means and variances must be calculated?
""")

    with st.expander("Answer"):
        st.markdown("Layer Norm normalizes over the `embedding` dimension (i.e. the number of means computed is `batch_size * sequence_length`).")


    st.markdown("""## 2. \*

Look at the [PyTorch documentation page](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for `LayerNorm`. Play around with a LayerNorm instance until you understand how it is used.

## 3.

Dropout is the second type of layer we've found that behaves differently in training and eval modes (can you remember what the first was?). Explain how dropout changes behaviour in training vs eval mode.
""")

    with st.expander("Answer"):
        st.markdown("""In training mode, a dropout layer will take each scalar in the input and randomly set it to 0 with probability $p$, then divide all other scalars by $(1-p)$  (for normalisation reasons). In eval mode, a dropout layer doesn't do this; none of the scalars are set to zero.""")

def section_7():
    st.markdown("""# Softmax and Activation Functions

## 1. \*

Prove that softmax is invariant under adding a constant scalar c to each dimension of the input.""")

    with st.expander("Answer"):
        st.markdown(r"""
$$
\begin{aligned}
(\sigma(\underline{z}+c))_i &=\frac{e^{z_i+c}}{\sum_j e^{z_j+c}} \\
&=\frac{e^{z_i} \times e^c}{\left(\sum_j e^{z_j}\right) \times e^c} \\
&=\frac{e^{z_i}}{\sum_j e^{z_j}} \\
&=\sigma(\underline{z})_i
\end{aligned}
$$

Intuition: the logits represent the "bits of evidence for a particular outcome", in some sense. Adding the same amount of evidence for each outcome doesn't raise one probability more than any other.
""")

    st.markdown(r"""

## 2. \**

In the softmax function, we sometimes use a parameter $T>0$ to denote temperature. We divide the vector by $T$ before applying exp then normalising. What effect will $T$ have on the output probabilities?
""")

    with st.expander("Answer"):
        st.markdown(r"""If $T$ is very small, the probabilities will tend to zero everywhere, except for the largest value which will tend to 1 (note - this is why the function is called softmax; it's a softened version of the max function!). In effect, the model is more confident.

If $T$ is very large, the probabilities will be pushed closer to uniform. In effect, the model is less confident.

Note - the name **temperature** comes from the fact that this parameter plays a similar role in **statistical mechanics**, in the formula for the [Boltzmann distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution).""")

    st.markdown("""
## 3.

Try computing the GELU function from [this paper](https://arxiv.org/pdf/1606.08415.pdf), and plotting it. Compare it to the GELU approximation from the same paper. Where do they diverge the most, and by how much?

## 4. 

Try also plotting the **Swish** function. What are the advantages you can see of Swish and GELU compared to ReLU and sigmoid?
""")

func_list = [section_home, section_1, section_2, section_3, section_4, section_5, section_6, section_7]

page_list = ["🏠 Home", "1️⃣ Transformers (general)", "2️⃣ Training", "3️⃣ Attention mechanism", "4️⃣ Tokenisation and embedding", "5️⃣ Positional encoding", "6️⃣ Layer Normalisation and Dropout", "7️⃣ Softmax and Activation Functions"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()
# for idx, section in enumerate(sections_selectbox):
#     func_list[idx]()
