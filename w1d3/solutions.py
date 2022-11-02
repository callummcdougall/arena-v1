from einops import repeat, rearrange
from fancy_einsum import einsum
import torch as t
from torch import nn
from typing import Optional

device = t.device("cuda" if t.cuda.is_available() else "cpu")
assert str(device) == "cuda"

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