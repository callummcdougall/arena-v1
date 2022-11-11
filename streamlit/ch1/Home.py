import streamlit as st

st.set_page_config(layout="wide")

import platform
is_local = (platform.processor() != "")
rootdir = "" if is_local else "streamlit/ch1/"

st.markdown("""
<style>
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
    color:red;
    white-space: pre-wrap !important;
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

# st.sidebar.markdown("""
# ## Table of Contents

# <ul class="contents">
#     <li><a class="contents-el" href="#about-this-page">About this page</a></li>
#     <li><a class="contents-el" href="#hints">Hints</a></li>
#     <li><a class="contents-el" href="#test-functions">Test functions</a></li>
#     <li><a class="contents-el" href="#tips">Tips</a></li>
#     <li><a class="contents-el" href="#support">Support</a></li>
# </ul>
# """, unsafe_allow_html=True)

st.image(rootdir + "images/headers/trans.png", width=320)

st.markdown("""
# Transformers

The transformer is an important neural network architecture used for language modeling.

In this chapter, you will learn all about transformers - how the attention mechanism works, how transformers are trained, and how they’ve managed to be the driving force behind language model progress of the last few years.

Some highlights from this chapter include:

* Building your own decoder-only transformer (which includes building a unidirectional attention block)
* Training your transformer on the corpus of Shakespeare text, to generate realistic output
* Assembling and loading weights into GPT-2, and sampling output from it
* Training a transformer to solve some classic LeetCode-style problems
""")