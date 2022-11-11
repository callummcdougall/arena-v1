import streamlit as st

st.set_page_config(layout="wide")

import platform
is_local = (platform.processor() != "")
rootdir = "" if is_local else "streamlit/ch0/"

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

st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#about-this-page">About this page</a></li>
    <li><a class="contents-el" href="#hints">Hints</a></li>
    <li><a class="contents-el" href="#test-functions">Test functions</a></li>
    <li><a class="contents-el" href="#tips">Tips</a></li>
    <li><a class="contents-el" href="#support">Support</a></li>
</ul>
""", unsafe_allow_html=True)

st.image(rootdir + "images/headers/pre.png", width=320)

st.markdown("""

# Prerequisites 

Before embarking on this curriculum, it is necessary to understand the basics of deep learning, including basic machine learning terminology, what neural networks are, and how to train them.

Some highlights from this chapter include:
* Building your own 1D and 2D convolution functions
* Building and loading weights into a Residual Neural Network, and finetuning it on a classification task
* Working with [weights and biases](https://wandb.ai/site) to optimise hyperparameters
* Implementing your own backpropagation mechanism

---

## About this page

This page was made using an app called Streamlit. It's hosted from the [ARENA GitHub repo](https://github.com/callummcdougall/arena-v1-streamlit). It provides a very simple way to display markdown, as well as more advanced features like interactive plots and animations. This is how the instructions for each day will be presented. You may also want to learn Streamlit in order to have a convenient way to present the results of your personal projects at the end of each week.

On the left, you can see a sidebar (or if it's collapsed, you will be able to see if you click on the small arrow in the top-left to expand it). This sidebar should show a page called `Home` (which is the page you're currently reading), as well as one for each of the different parts of today's exercises.""")

st.info("""
Note - these exercises form different sections of the day, rather than corresponding to different days. At the start of each exercise, I've included an estimated completion time. This should be taken with a pinch of salt (you might prefer to go at different speeds, or be more/less comfortable with certain sections). But if you find yourself going well outside this estimate, then it's probably a sign that you should be more willing to ask for help (either by sending a message in the `#technical-questions` Slack, or sending a direct message to Callum).
""")

st.markdown("""If you want to change to dark mode, you can do this by clicking the three horizontal lines in the top-right, then navigating to Settings → Theme.

## Hints

There will be occasional hints throughout the document, for when you're having trouble with a certain task but you don't want to read the solutions. Click on the expander to reveal the solution in these cases. Below is an example of what they'll look like:""")

with st.expander("Help - I'm stuck on a particular problem."):
    st.markdown("Here is the answer!")

st.markdown("""Always try to solve the problem without using hints first, if you can.

## Test functions

Most of the blocks of code will also come with test functions. These are imported from python files with names such as `w0d1/utils.py`. You should make sure these files are in your working directory while you're writing solutions. One way to do this is to clone the [main GitHub repo](https://github.com/callummcdougall/arena-v1) into your working directory, and run it there. When we decide exactly how to give participants access to GPUs, we might use a different workflow, but this should suffice for now. Make sure that you're getting the most updated version of utils at the start of every day (because changes might have been made), and keep an eye out in the `#errata` channel for mistakes which might require you to change parts of the test functions.

## Tips

* To get the most out of these exercises, make sure you understand why all of the assertions should be true, and feel free to add more assertions.
* If you're having trouble writing a batched computation, try doing the unbatched version first.
* If you find these exercises challenging, it would be beneficial to go through them a second time so they feel more natural.

## Support

If you ever need help, you can send a message on the ARENA Slack channel `#technical-questions`. You can also reach out to a TA (e.g. Callum) if you'd like a quick videocall to talk through a concept or a problem that you've been having, although there might not always be someone available.

You can also read the solutions by downloading them from the [GitHub](https://github.com/callummcdougall/arena-v1). However, ***this should be a last resort***. Really try and complete the exercises as a pair before resorting to the solutions. Even if this involves asking a TA for help, this is preferable to reading the solutions. If you do have to read the solutions, then make sure you understand why they work rather than just copying and pasting. 

At the end of each day, it can be beneficial to look at the solutions. However, these don't always represent the optimal way of completing the exercises; they are just how the author chose to solve them. If you think you have a better solution, we'd be really grateful if you could send it in, so that it can be used to improve the set of exercises for future ARENA iterations.

Happy coding!
""")