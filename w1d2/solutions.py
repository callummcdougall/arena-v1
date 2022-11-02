
from einops import repeat
from fancy_einsum import einsum
import torch as t

def attention(Q, K, V):
    batch, seq_len, hidden_size = Q.shape

    # Calculate attention scores via matrix operation
    attention_scores = einsum("batch seqQ hidden, batch seqK hidden -> batch seqQ seqK", Q, K) / t.sqrt(hidden_size)

    # Take softmax so that each row is a probability (after dividing by scale factor root(d_k))
    attention_probabilities = attention_scores.softmax(dim=-1)

    attention_values = einsum("batch seqQ seqK", "batch seqK hidden -> batch seqQ hidden", attention_probabilities, V)

    return attention_values
    

def masked_attention(Q, K, V):
    batch, seq_len, hidden_size = Q.shape
    attention_scores = einsum("batch seqQ hidden, batch seqK hidden -> batch seqQ seqK", Q, K) / t.sqrt(hidden_size)

    # Note we don't need to add batch and nheads to the mask, for broadcasting reasons
    Q_idx = repeat(t.arange(seq_len), "seqQ -> seqQ seqK", seqK=seq_len)
    K_idx = repeat(t.arange(seq_len), "seqK -> seqQ seqK", seqQ=seq_len)
    # Any index positions with q<k should be masked (this represents tokens "getting info from the future")
    mask = (Q_idx < K_idx).to(t.bool)
    attention_scores = t.where(mask, t.tensor(1e-6), attention_scores)

    # Take softmax over the key dimension (i.e. each query index corresponds to a probability vector)
    attention_probabilities = attention_scores.softmax(dim=-1)

    # Get attention values by taking convex combination of value vectors
    attention_values = einsum("batch seqQ seqK", "batch seqK hidden -> batch seqQ hidden", attention_probabilities, V)

    return attention_values