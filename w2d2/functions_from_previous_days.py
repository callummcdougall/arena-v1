# %%
import torch as t
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm_notebook
import torchinfo
from fancy_einsum import einsum
from typing import Optional, Union, List
import numpy as np
import functools
from einops import rearrange, repeat
from dataclasses import dataclass
import plotly.express as px


# %%

def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """
    B, C, W = x.shape
    output = x.new_full(size=(B, C, left + W + right), fill_value=pad_value)
    output[..., left : left + W] = x
    # Note - you can't use `left:-right`, because `right` could be zero.
    return output

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    B, C, H, W = x.shape
    output = x.new_full(size=(B, C, top + H + bottom, left + W + right), fill_value=pad_value)
    output[..., top : top + H, left : left + W] = x
    return output

def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    """Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    
    x_padded = pad1d(x, left=padding, right=padding, pad_value=0)
    
    batch, in_channels, width = x_padded.shape
    out_channels, in_channels_2, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"
    output_width = 1 + (width - kernel_width) // stride
    # note, we assume padding is zero in the formula here, because we're working with input which has already been padded
    
    xsB, xsI, xsWi = x_padded.stride()
    wsO, wsI, wsW = weights.stride()
    
    x_new_shape = (batch, in_channels, output_width, kernel_width)
    x_new_stride = (xsB, xsI, xsWi * stride, xsWi)
    # Explanation for line above:
    #     we need to multiply the stride corresponding to the `output_width` dimension
    #     because this is the dimension that we're sliding the kernel along
    x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)
    
    return einsum("B IC OW wW, OC IC wW -> B OC OW", x_strided, weights)

IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    """Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    """

    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    
    x_padded = pad2d(x, left=padding_w, right=padding_w, top=padding_h, bottom=padding_h, pad_value=0)
    
    batch, in_channels, height, width = x_padded.shape
    out_channels, in_channels_2, kernel_height, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"
    output_width = 1 + (width - kernel_width) // stride_w
    output_height = 1 + (height - kernel_height) // stride_h
    
    xsB, xsIC, xsH, xsW = x_padded.stride() # B for batch, IC for input channels, H for height, W for width
    wsOC, wsIC, wsH, wsW = weights.stride()
    
    x_new_shape = (batch, in_channels, output_height, output_width, kernel_height, kernel_width)
    x_new_stride = (xsB, xsIC, xsH * stride_h, xsW * stride_w, xsH, xsW)
    
    x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)
    
    return einsum("B IC OH OW wH wW, OC IC wH wW -> B OC OH OW", x_strided, weights)

def maxpool2d(x: t.Tensor, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0
) -> t.Tensor:
    """Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    """

    if stride is None:
        stride = kernel_size
    stride_height, stride_width = force_pair(stride)
    padding_height, padding_width = force_pair(padding)
    kernel_height, kernel_width = force_pair(kernel_size)
    
    x_padded = pad2d(x, left=padding_width, right=padding_width, top=padding_height, bottom=padding_height, pad_value=-t.inf)
    
    batch, channels, height, width = x_padded.shape
    output_width = 1 + (width - kernel_width) // stride_width
    output_height = 1 + (height - kernel_height) // stride_height
    
    xsB, xsC, xsH, xsW = x_padded.stride()
    
    x_new_shape = (batch, channels, output_height, output_width, kernel_height, kernel_width)
    x_new_stride = (xsB, xsC, xsH * stride_height, xsW * stride_width, xsH, xsW)
    
    x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)
    
    output = t.amax(x_strided, dim=(-1, -2))
    return output






class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Call the functional version of maxpool2d."""
        return maxpool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])


class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0.0))


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape
        
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        
        shape_left = shape[:start_dim]
        shape_middle = functools.reduce(lambda x, y: x*y, shape[start_dim : end_dim+1])
        shape_right = shape[end_dim+1:]
        
        new_shape = shape_left + (shape_middle,) + shape_right
        
        return t.reshape(input, new_shape)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        sf = 1 / np.sqrt(in_features)
        
        weight = sf * (2 * t.rand(out_features, in_features) - 1)
        self.weight = nn.Parameter(weight)
        
        if bias:
            bias = sf * (2 * t.rand(out_features,) - 1)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        x = einsum("... in_features, out_features in_features -> ... out_features", x, self.weight)
        if self.bias is not None: x += self.bias
        return x

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        kernel_height, kernel_width = force_pair(kernel_size)
        sf = 1 / np.sqrt(in_channels * kernel_width * kernel_height)
        weight = sf * (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1)
        self.weight = nn.Parameter(weight)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d you wrote earlier."""
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        for i, mod in enumerate(modules):
            self.add_module(str(i), mod)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            if mod is not None:
                x = mod(x)
        return x

class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))
        
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        
        # Calculating mean and var over all dims except for the channel dim
        if self.training:
            # Using keepdim=True so we don't have to worry about broadasting them with x at the end
            mean = t.mean(x, dim=(0, 2, 3), keepdim=True)
            var = t.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
            # Updating running mean and variance, in line with PyTorch documentation
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        else:
            mean = rearrange(self.running_mean, "channels -> 1 channels 1 1")
            var = rearrange(self.running_var, "channels -> 1 channels 1 1")
        
        # Rearranging these so they can be broadcasted (although there are other ways you could do this)
        weight = rearrange(self.weight, "channels -> 1 channels 1 1")
        bias = rearrange(self.bias, "channels -> 1 channels 1 1")
        
        return ((x - mean) / t.sqrt(var + self.eps)) * weight + bias

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])






# ==================== TRANSFORMER ARCHITECTURE (NOT REQUIRING NEW MODULES) ====================

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

def multihead_masked_attention(Q, K, V, num_heads):

    device = Q.device

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




# ==================== DATA & TRAINING ====================

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

def train(model, optimizer, loss_fn, trainloader, epochs, dataset_name=None, plot_loss=True):

    device = next(model.parameters()).device

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









# ==================== SAMPLING ====================

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