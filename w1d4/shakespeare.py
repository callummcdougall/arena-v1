# %%

# ============================= IMPORTS =============================

import os
# This makes a certain kind of error message more legible
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import plotly.express as px
import torch as t
import re
import numpy as np
import transformers
from torch import nn, optim
import random
import pandas as pd
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, reduce, repeat
from fancy_einsum import einsum
from typing import Optional, Callable, Any, List, Dict, Union
from tqdm.notebook import tqdm_notebook
from IPython.display import display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = t.device("cuda" if t.cuda.is_available() else "cpu")
assert str(device) == "cuda"

# %%



# ============================= TRANSFORMER ARCHITECTURE =============================

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
    print_param_count: bool = True

# %%

class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        # Defining our positional encoding array, with `max_seq_len` rows
        # This is an advantage of using sinusoidal encoding: we can easily expand to sequences of greater length without adding more learned params
        angles = t.outer(t.arange(max_seq_len), 1 / 10000 ** (2 * t.arange(embedding_dim//2) / embedding_dim))
        pe = t.zeros((max_seq_len, embedding_dim))
        pe[:, ::2] = t.sin(angles)
        pe[:, 1::2] = t.cos(angles)
        # Register array as a buffer, rather than parameter (we don't want it to be updated by gradient descent)
        self.register_buffer('pe', pe)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq_len, embedding_dim)
        """
        batch, seq_len, embedding_dim = x.shape
        # We slice the positional encoding, so it's the same shape as x
        # This is equivalent to just using an nn.Embedding, but having the input be t.arange(seq_len)
        return x + self.pe[:seq_len, :] # type: ignore

# %%

def multihead_masked_attention(Q, K, V, num_heads):

    # Rearrange Q, K and V to separate the `headsize` dimension (because this is the one we take the inner product over)
    q = rearrange(Q, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
    k = rearrange(K, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
    v = rearrange(V, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)

    # Calculate attention scores as inner product of q and k, and divide by sqrt(headsize)
    batch, seq_len, nheads, headsize = q.shape
    attention_scores = einsum("batch seqQ nheads headsize, batch seqK nheads headsize -> batch nheads seqQ seqK", q, k) / (headsize ** 0.5)

    # Create the attention mask
    # Note we don't need to add batch and nheads, for broadcasting reasons
    # Also note you could do this with much less code using e.g. t.triu(t.ones(...)), but this way is more explicit
    q_idx = repeat(t.arange(seq_len), "seqQ -> seqQ seqK", seqK=seq_len)
    k_idx = repeat(t.arange(seq_len), "seqK -> seqQ seqK", seqQ=seq_len)
    # Any index positions with q<k should be masked (this prevents tokens "reading info from the future")
    mask = (q_idx >= k_idx).to(device)
    neg_inf = t.tensor(-1e6, dtype=attention_scores.dtype, device=device)
    attention_scores = t.where(mask, attention_scores, neg_inf)

    # Take softmax over the key dimension (i.e. each query index has a corresponding probability distribution over tokens in the sequence)
    attention_probabilities = attention_scores.softmax(dim=-1)

    # Get attention values by taking convex combination of value vectors according to the attention probabilities
    attention_values = einsum("batch nheads seqQ seqK, batch seqK nheads headsize -> batch seqQ nheads headsize", attention_probabilities, v)

    # Rearrange to combine the nheads and headsize dimensions
    return rearrange(attention_values, "batch seqQ nheads headsize -> batch seqQ (nheads headsize)")


# %%

class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int, head_size: Optional[int] = None):
        """
        Adding option to override head_size (defaults to hidden_size / num_heads otherwise)
        """
        super().__init__()
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads if head_size is None else head_size
        
        # Note that these weight matrices are usually called projections and defined as linear layers without bias, but they are 
        # still implemented with bias in some papers.
        self.W_QKV = nn.Linear(hidden_size, 3*num_heads*self.head_size)
        self.W_O = nn.Linear(num_heads*self.head_size, hidden_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        # Computationally faster to apply W_QKV on x before splitting
        QKV = self.W_QKV(x)
        Q, K, V = t.split(QKV, self.num_heads*self.head_size, dim=-1)

        masked_attention_values = multihead_masked_attention(Q, K, V, num_heads=self.num_heads)

        output = self.W_O(masked_attention_values)

        return output


# %%

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))

# %%

class DecoderBlock(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MultiheadMaskedAttention(config.hidden_size, config.num_heads)
        self.mlp = MLP(config)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x + self.ln1(self.attn(x))
        x = x + self.ln2(self.mlp(x))
        return x

        
class DecoderOnlyTransformer(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.positional_encoding = PositionalEncoding(config.max_seq_len, config.hidden_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.Sequential(*[DecoderBlock(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Function to print a dataframe visualising parameter count (this can be omitted, but it's pretty useful!)
        if config.print_param_count:
            print(f"Total params = {sum([param.numel() for param in self.parameters()])}")
            with pd.option_context("display.max_rows", 1000):
                df = pd.DataFrame([
                    {"name": name, "shape": tuple(param.shape), "num params": param.numel()}
                    for name, param in self.named_parameters()
                ])
                display(df.style.background_gradient(cmap="viridis", subset=["num params"], gmap=np.log(df["num params"])))

        
    def forward(self, x: t.Tensor) -> t.Tensor:
        # If x has no batch dimension, give it one (this means the transformer can also be run on 1D inputs with no batch dimension)
        if len(x.shape) == 1:
            x = rearrange(x, "seq -> 1 seq")
        # Apply token embedding before positional encoding (this is easier, because then PE can just be added to x)
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)
        x = einsum("batch seq hidden, vocab hidden -> batch seq vocab", x, self.token_embedding.weight)
        return x
# %%



# ============================= REVERSED SEQUENCES =============================

class ReverseDataset(Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.vocab_size = 10 # digits from 0 to 9 inclusive
        self.size = 10 ** seq_len # so that each seq appears once in the dataset (in expectation)

    def __len__(self):
        # This is what is returned when you call len(dataset)
        # And it's what PyTorch uses to construct the dataset when initialised
        return self.size

    def __getitem__(self, idx):
        # Rather than randomising, could also generate every single sequence
        seq = t.randint(self.vocab_size, size=(self.seq_len,), dtype=t.long)
        seq_reversed = seq.flip(-1)
        return seq, seq_reversed

# Create dataset for training
seq_len = 6
trainset = ReverseDataset(seq_len=seq_len)

# %%
batch_size = 1024
trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)

# %%
config = TransformerConfig(
    num_layers = 2,
    num_heads = 6,
    vocab_size = trainset.vocab_size,
    hidden_size = 96,
    max_seq_len = trainset.seq_len,
)

model = DecoderOnlyTransformer(config).to(device).train()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 2

# %%

def train(model, optimizer, loss_fn, trainloader, epochs, dataset_name=None, plot_loss=True):

    loss_list = []

    for epoch in range(epochs):
        
        progress_bar = tqdm_notebook(trainloader)
        
        for (x, y) in progress_bar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            
            logits = model(x)
            # logits dimensions are (batch, seq, digits), but we care about probabilities for each digit
            # so we need to reshape into (batch * seq, digits)
            loss = loss_fn(rearrange(logits, "b s d -> (b s) d"), y.flatten())
            loss.backward()

            optimizer.step()
            
            progress_bar.set_description(f"epoch = {epoch+1}, loss = {loss.item():.4f}")

            loss_list.append(loss.item())

    # Function to plot the loss over epochs
    if plot_loss:
        fig = px.line(
            y=loss_list, 
            template="simple_white", 
            labels={
                "x": "No. batches seen", 
                "y": str(loss_fn).replace("()", "") # This gets a name like "CrossEntropyLoss" from the loss function
            }, 
            title=f"Training loss on {dataset_name} dataset" if dataset_name is not None else "Training loss"
        )
        # This next bit of code plots vertical lines corresponding to the epochs
        if epochs > 1:
            for idx, epoch_start in enumerate(np.linspace(0, len(loss_list), epochs, endpoint=False)):
                fig.add_vline(x=epoch_start, line_width=3, line_dash="dash", annotation_text=f"Epoch {idx}", annotation_position="top right")
        fig.show()
    
    return model
# %%

model = train(model, optimizer, loss_fn, trainloader, epochs, "ReversedDigits")
# With this model and parameters, I found loss dropping to about 1.17 after second epoch

# %%

model.eval()
seq = t.randint(10, size=(6,), dtype=t.long, device=device)
seq_reversed = seq.flip(-1)
logits = model(seq)
prediction = logits.argmax(dim=-1).squeeze()
print("prediction:", prediction)
print("answer:", seq_reversed)
t.testing.assert_close(seq_reversed[-3:], prediction[-3:])
# As expected, model is getting the first three digits wrong, but the last three incorrect (so attention masking is working)

# %%




# ============================= SHAKESPEARE =============================

# Load the text data
with open("100-0.txt", encoding="utf-8") as file:
    text = file.read()
    words = re.split(r"\b", text)

# %%

class WordsDataset(Dataset):
    def __init__(self, words, seq_len, fraction):
        """
        `fraction` is so we can scale down the amount of training that we do (otherwise it's a big dataset!). 
        
        This parameter will change the total length, and hence changes epoch duration (from hours to minutes).
        """
        self.fraction = fraction
        self.seq_len = seq_len
        self.words = words
        # Max len is less than # words, because we need to take a slice of tokens for getitem
        self.max_len = len(self.words) - (self.seq_len + 1)
        self.vocab_size = len(set(words))
        self.words_to_token_idx = {word: idx for (idx, word) in enumerate(sorted(set(words)))}
        self.token_idx_to_words = {idx: word for (word, idx) in self.words_to_token_idx.items()}
        self.tokens = t.tensor([self.words_to_token_idx[word] for word in words]).to(dtype=t.long)

    def __len__(self):
        return int(self.max_len * self.fraction)

    def __getitem__(self, idx):
        # Given tokens (t_1, ..., t_n), we want to predict (t_2, ..., t_n+1)
        # This is actually n separate instances of task "predict j+1th token from first j tokens", for 1<=j<=n
        x_and_y = self.tokens[idx: idx + self.seq_len + 1]
        x = x_and_y[:-1]
        y = x_and_y[1:]
        return x, y

max_seq_len = 48
trainset = WordsDataset(words=words, seq_len=max_seq_len, fraction=0.02)

batch_size = 32
trainloader = DataLoader(trainset, shuffle=True, pin_memory=True, batch_size=batch_size)

# Create a tokenizer, so I can do things like tokenizer.encode(initial_text) and tokenizer.decode(list_of_ids)
# Also using the `return_tensors` argument of the encode method, just like gpt's tokenizer does
# This object is optional though, you could just use `self.words_to_token_idx` directly from dataset
class WordsTokenizer():
    def __init__(self, wordsdataset: WordsDataset):
        self.words_to_token_idx = wordsdataset.words_to_token_idx
        self.token_idx_to_words = wordsdataset.token_idx_to_words

    def encode(self, initial_text: str, return_tensors: Optional[str] = None) -> Union[list, np.ndarray, t.Tensor]:
        list_of_strings = [s for s in re.split(r"\b", initial_text) if len(s) > 0]
        tensors_list = [self.words_to_token_idx[s] for s in list_of_strings]
        if return_tensors is None:
            return tensors_list
        elif return_tensors == "pt":
            return t.tensor(tensors_list)
        elif return_tensors == "np":
            return np.array(tensors_list)
        else:
            raise Exception("Unexpected value for `return_tensors`.")

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        return ''.join([self.token_idx_to_words[int(token)] for token in list_of_ids])

tokenizer = WordsTokenizer(trainset)

# %%
config = TransformerConfig(
    num_layers = 8,
    num_heads = 8,
    vocab_size = trainset.vocab_size,
    hidden_size = 512,
    max_seq_len = trainset.seq_len,
    dropout = 0.1,
    layer_norm_epsilon = 1e-05
)

model = DecoderOnlyTransformer(config).to(device).train()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 1

# %%

model = train(model, optimizer, loss_fn, trainloader, epochs, "WordsDataset")
# With this model and parameters, I had loss down to about 1.7 by the end of one epoch

# %%

# import the sampling methods
from solutions import *

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
    model: DecoderOnlyTransformer,
    tokenizer: WordsTokenizer,
    initial_text: str,
    max_tokens_generated=30,
    **kwargs # kwargs are for params like temperature, top_k, etc
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    # Note - an alternative to model.eval() is to use the @t.inference_mode() decorator for this whole function.
    model.eval()
    input_ids: list = tokenizer.encode(initial_text) # type: ignore
    generated = []
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.long, device=device)
        new_input_ids_window = new_input_ids[-min(max_seq_len, new_input_ids.shape[0]):]
        logits = model(new_input_ids_window)[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)

# Note, some initial text strings might not work because they weren't present in the text you used for training
initial_text = "turn down for what"

text_output = sample_tokens(model, tokenizer, initial_text, max_tokens_generated=100, temperature=1.0, top_k=10)

print(text_output)

# Result:

# turn down for what you do you think,
# That take the last, of many, which is so much I
# As this blows along than my life thou say’st, which makes thy hand,
# Thou wilt be given, or more
# Entitled in thy great world’s fresh blood will,
# To answer th’ alluring countenance, beauty 

# %%
