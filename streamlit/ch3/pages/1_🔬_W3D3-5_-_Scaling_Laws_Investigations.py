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
    st.markdown("""
## 1️⃣ Reading

This section includes a few seminal papers on scaling laws. We strongly recommend that you read at least section 1 of the first paper. Which other papers you read will depend largely on your own interests, although you're encouraged to try at least one of them.

## 2️⃣ Suggested exercises

In this section, we discuss a few exercises which you might want to attempt. Most of them are based in some way on the material you will have read about in section 1. Like the exercises at the end of the optimisation week, these are designed to be challenging and open-ended.

---

We haven't allocated a large section of the course to this section, because it's less suitable for structured exercises. You may prefer to prioritise doing more work on the transformers and optimisation sections of the course, and return to this material later on (or after the programme finishes), if you prefer.
""")

def section1():
    st.markdown("""
This section includes a few seminal papers on scaling laws. We strongly recommend that you read at least section 1 of the first paper. Which other papers you read will depend largely on your own interests, although you're encouraged to try at least one of them.

## Recommended reading 

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - A study of how transformer language models scale with model size, dataset size and compute. Section 1 has a good summary of the main results.

## Optional reading

- [Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701) - A study of how transformers trained on different data distributions scale, introducing the idea of the irreducible loss.
- [Scaling Laws for Transfer](https://arxiv.org/abs/2102.01293) - A study of how pre-training on natural language improves fine-tuning on Python. In the low-fine-tuning-data regime, pre-training acts as an effective multiplier on the amount of fine-tuning data.
- [Unified Scaling Laws for Routed Language Models](https://arxiv.org/abs/2202.01169) - Scaling laws for MOEs
- [Scaling Scaling Laws with Board Games](https://arxiv.org/abs/2104.03113) - Scaling laws for AlphaZero on Hex
- [Chinchilla](https://arxiv.org/abs/2203.15556) - A correction to the original scaling laws paper: parameter count scales linearly with token budget for compute-optimal models, not ~quadratically. The difference comes from using a separately-tuned learning rate schedule for each token budget, rather than using a single training run to measure performance for every token budget. This highlights the importance of hyperparameter tuning for measuring scaling law exponents.
- [The Scaling Hypothesis](https://www.gwern.net/Scaling-hypothesis) - A post by Gwern, discussing meta-learning, scaling, and its implications
""")

def section2():
    st.markdown(r"""
In this section, we discuss a few exercises which you might want to attempt. Most of them are based in some way on the material you will have read about in section 1. Like the exercises at the end of the optimisation week, these are designed to be challenging and open-ended.

## Perform your own study of scaling laws for MNIST

* Write a script to train a small CNN on MNIST, or find one you have written previously.
* Training for a single epoch only, vary the model size and dataset size. For the model size, multiply the width by powers of $\sqrt{2}$ (rounding if necessary - the idea is to vary the amount of compute used per forward pass by powers of $2$). For the dataset size, multiply the fraction of the full dataset used by powers of $2$ (i.e. $1, 1/2, 1/4, ...$). To reduce noise, use a few random seeds and always use the full validation set.
* The learning rate will need to vary with model size. Either tune it carefully for each model size, or use the rule of thumb that for Adam, the learning rate should be proportional to the initialization scale, i.e. $1/\sqrt{\text{fan\_in}}$ for the standard Kaiming He initialization (which is what PyTorch generally uses by default).
* Plot the amount of compute used (on a log scale) against validation loss. The compute-efficient frontier should follow an approximate power law (straight line on a log scale).
* How does validation accuracy behave?
* Study how the compute-efficient model size varies with compute. This should also follow an approximate power law. Try to estimate its exponent.
    * The [Chinchilla paper]() highlighted the importance of hyperparameter tuning for measuring scaling law exponents. How does your exponent change if you first use `wandb` sweeps to find the optimal hyperparameters for the different testing regimes? You might want to go back to the start of this week and read up on [learning rate schedulers](https://arena-ch2.streamlit.app/W3D1_-_Optimiser_Exercises#why-learning-rate-schedules), since this is also an important thing to optimise.
* Repeat your entire experiment with 20% dropout to see how this affects the scaling exponents.
    * Can you think of a reason why dropout might increase / decrease model efficiency? How much does this depend on th exact details of how dropout is implemented? See [this Google doc](https://docs.google.com/document/d/1J2BX9jkE5nN5EA1zYRN0lHhdCf1YkiFERc_nwiYqCOA/edit#heading=h.a2552o2358pi) for some ideas. 
    * You can also try using the `Dropout` module that you may have built earlier in this programme (if not, you can go back and build it [here](https://arena-ch1.streamlit.app/W2D2-5_-_Further_Investigations#nn-dropout)). You could try a few different ways of implementing dropout, and benchmark them.
""")

    with st.expander("Help - I'm not sure how to estimate the amount of compute."):
        st.markdown("""[This Google doc](https://docs.google.com/document/d/1J2BX9jkE5nN5EA1zYRN0lHhdCf1YkiFERc_nwiYqCOA/edit#) might help. Section 1 gives a useful overview of how to estimate the compute of deep learning models by counting the total number of operations performed. In particular, the first two rows of the table on page 5 should help you, if you're just using a CNN with fully connected and convolutional layers.

If you're stuck on this, then as a first pass you can just try measuring parameter count (you'll be able to calculate this easily from your model, by summing over `model.parameters()`).
""")

    st.info("""
As an extra challenge, you can try to derive scaling laws for your ResNet or your transformer. [This LessWrong post](https://www.lesswrong.com/posts/b3CQrAo2nufqzwNHF/how-to-train-your-transformer) outlines the details of a project to train a transformer to predict human moves in chess games, using data from [Lichess](https://database.lichess.org/). The post suggests using Colab to access a good GPU, but you might also like to return to this section once you've learned how to set up and run your models on a GPU (if you're planning on taking that track next week).
""")

func_list = [section_home, section1, section2]

page_list = ["🏠 Home", "1️⃣ Reading", "2️⃣ Suggested exercises"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

with st.sidebar:

    radio = st.radio("Section", page_list)

    st.markdown("---")

func_list[page_dict[radio]]()
# for idx, section in enumerate(sections_selectbox):
#     func_list[idx]()
