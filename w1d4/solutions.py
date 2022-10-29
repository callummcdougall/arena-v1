import torch as t

def greedy_search(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    """
    out = logits.argmax().item()
    assert isinstance(out, int)
    return out

def sample_basic(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    """
    distribution = t.distributions.categorical.Categorical(logits=logits)
    out = distribution.sample().item()
    assert isinstance(out, int)
    return out

def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    """
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    assert temperature > 0
    return logits / temperature

def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    """
    input_ids: shape (seq, )
    logits: shape (vocab_size, )
    Return: shape (vocab_size, )
    """
    (vocab_size,) = logits.shape
    id_freqs = t.bincount(input_ids, minlength=vocab_size)
    return logits - freq_penalty * id_freqs

def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    """
    top_logits, top_idx = t.topk(logits, top_k)
    idx = t.distributions.categorical.Categorical(logits=top_logits).sample()
    return top_idx[idx].item()

def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    Return: a sampled token
    """
    logits_sorted, indices = logits.sort(descending=True, stable=True)
    cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
    n_keep = t.searchsorted(cumul_probs, top_p, side="right").item() + 1
    n_keep = max(n_keep, min_tokens_to_keep)
    keep_idx = indices[:n_keep]
    keep_logits = logits[keep_idx]
    sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
    return keep_idx[sample].item()

