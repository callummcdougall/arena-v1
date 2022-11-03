# %%
# ============================= IMPORTS =============================

import os
# This makes a certain kind of error message more legible
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch as t
import transformers
from torch import nn
from dataclasses import dataclass
from einops import rearrange, repeat
from fancy_einsum import einsum

import utils

device = t.device("cuda" if t.cuda.is_available() else "cpu")
assert str(device) == "cuda"

# %%
# ============================= CONFIG =============================

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

config = TransformerConfig(
    num_layers = 12,
    num_heads = 12,
    vocab_size = 50257,
    hidden_size = 768,
    max_seq_len = 1024,
    dropout = 0.1,
    layer_norm_epsilon = 1e-05,
    print_param_count = False
)

# %%
# ============================= TRANSFORMER ARCHITECTURE =============================

class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, config):
        """
        Adding option to override head_size (defaults to hidden_size / num_heads otherwise)
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        
        # Note that these weight matrices are usually called projections and defined as linear layers without bias, but they are 
        # still implemented with bias in some papers.
        self.W_QKV = nn.Linear(self.hidden_size, 3*self.num_heads*self.head_size)
        self.W_O = nn.Linear(self.num_heads*self.head_size, self.hidden_size)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        # Computationally faster to apply W_QKV on x before splitting
        QKV = self.W_QKV(x)
        Q, K, V = t.split(QKV, self.num_heads*self.head_size, dim=-1)

        masked_attention_values = self.multihead_masked_attention(Q, K, V, num_heads=self.num_heads)

        output = self.W_O(masked_attention_values)

        return output

    # Now moving this function into a class method, so it can refer to the dropout layers
    def multihead_masked_attention(self, Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int) -> t.Tensor:

        q = rearrange(Q, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
        k = rearrange(K, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
        v = rearrange(V, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)

        batch, seq_len, nheads, headsize = q.shape
        attention_scores = einsum("batch seqQ nheads headsize, batch seqK nheads headsize -> batch nheads seqQ seqK", q, k) / (headsize ** 0.5)

        q_idx = repeat(t.arange(seq_len), "seqQ -> seqQ seqK", seqK=seq_len)
        k_idx = repeat(t.arange(seq_len), "seqK -> seqQ seqK", seqQ=seq_len)
        mask = (q_idx >= k_idx).to(device)
        neg_inf = t.tensor(-1e6, dtype=attention_scores.dtype, device=device)
        attention_scores = t.where(mask, attention_scores, neg_inf)

        attention_probabilities = attention_scores.softmax(dim=-1)
        # first dropout
        attention_probabilities = self.dropout1(attention_probabilities)

        attention_values = einsum("batch nheads seqQ seqK, batch seqK nheads headsize -> batch seqQ nheads headsize", attention_probabilities, v)

        out = rearrange(attention_values, "batch seqQ nheads headsize -> batch seqQ (nheads headsize)")

        # second dropout
        return self.dropout2(out)


class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class GPTBLock(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MultiheadMaskedAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

        
class GPT(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_encoding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.Sequential(*[GPTBLock(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Function to print a dataframe visualising parameter count (this can be omitted, but it's pretty useful!)
        if config.print_param_count:
            utils.print_param_count(self)
            
    def forward(self, x: t.Tensor) -> t.Tensor:
        # If x has no batch dimension, give it one (this means the transformer can also be run on 1D inputs with no batch dimension)
        if len(x.shape) == 1:
            x = rearrange(x, "seq -> 1 seq")
        # Apply token and positional embedding
        posn = t.arange(x.shape[1], device=x.device)
        x = self.token_embedding(x) + self.positional_encoding(posn)
        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)
        x = einsum("batch seq hidden, vocab hidden -> batch seq vocab", x, self.token_embedding.weight)
        return x


# %%
# ============================= INITIALISING MODELS =============================

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

my_gpt = GPT(config).to(device).train()
gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").to(device).train()

utils.print_param_count(my_gpt, gpt)

# %%
# ============================= LOADING WEIGHTS =============================

def copy_weights(my_gpt: GPT, gpt) -> GPT:
    '''Copy over the weights of `gpt` to your gpt implementation.'''

    # Here we use named params not state dict, because gpt doesn't have any buffers we care about
    # (I think all its buffers are attention masks)
    my_gpt_dict = dict(my_gpt.named_parameters())
    gpt_dict = dict(gpt.named_parameters())
    
    # Check the number of params/buffers is correct
    assert len(my_gpt_dict) == len(gpt_dict), "Number of layers is wrong. Have you done the prev step correctly?"
    
    # Initialise an empty dictionary to store the correct key-value pairs
    state_dict = {}
    
    for (my_param_name, my_param), (name, param) in zip(my_gpt_dict.items(), gpt_dict.items()):
        # Sometimes params are transposed
        if len(my_param.shape) == 2 and my_param.shape == param.T.shape:
            state_dict[my_param_name] = param.T
            # print(f"Copied params.T: {name} -> {my_param_name}")
        elif my_param.shape == param.shape:
            state_dict[my_param_name] = param
            # print(f"Copied params:   {name} -> {my_param_name}")
        else:
            raise Exception(f"Parameter shapes don't match: {my_param.shape} vs {param.shape}")

    if set(state_dict.keys()) != set(my_gpt.state_dict().keys()):
        raise Exception("State dicts don't match.")
    
    my_gpt.load_state_dict(state_dict)
    
    return my_gpt

my_gpt = copy_weights(my_gpt, gpt)

# %%



# %%
# ============================= TESTING =============================

utils.test_load_pretrained_weights(gpt, tokenizer)

utils.test_load_pretrained_weights(my_gpt, tokenizer)
