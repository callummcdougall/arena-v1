# %%

import utils
from functions_from_previous_days import *

device = t.device("cuda" if t.cuda.is_available() else "cpu")
assert str(device) == "cuda"

# %%

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(t.randn(num_embeddings, embedding_dim))

    def forward(self, x: t.LongTensor) -> t.Tensor:
        '''For each integer in the input, return that row of the embedding.
        '''
        return self.weight[x]

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"

utils.test_embedding(Embedding)

# %%

class GELU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return 0.5 * x * (1 + t.tanh((2 / t.pi) * (x + 0.044715 * x**3)))

# utils.plot_gelu(GELU)

# %%

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-05, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(t.ones(normalized_shape))
            self.bias = nn.Parameter(t.zeros(normalized_shape))

    def forward(self, x: t.Tensor) -> t.Tensor:
        assert len(self.normalized_shape) <= len(x.shape)

        dims = tuple(range(len(x.shape)-len(self.normalized_shape), len(x.shape)))

        mean = x.mean(dim=dims, keepdims=True)
        var = x.var(dim=dims, unbiased=False, keepdims=True)

        x = (x - mean) / ((var + self.eps) ** 0.5)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x

utils.test_layernorm_mean_1d(LayerNorm)
utils.test_layernorm_mean_2d(LayerNorm)
utils.test_layernorm_std(LayerNorm)
utils.test_layernorm_exact(LayerNorm)
utils.test_layernorm_backward(LayerNorm)


# %%

class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        assert 0 <= p < 1
        self.p = p

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training:
            mask = (t.rand(size=x.shape) < self.p).to(x.device)
            return t.where(mask, 0.0, x / (1 - self.p))
        else:
            return x

utils.test_dropout_eval(Dropout)
utils.test_dropout_training(Dropout)

# %%

# =============================== TRANSFORMER REIMPLEMENTATION, WITH OWN MODULES ===============================

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


class MultiheadMaskedAttention(nn.Module):
    W_QKV: Linear
    W_O: Linear

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
        self.W_QKV = Linear(hidden_size, 3*num_heads*self.head_size)
        self.W_O = Linear(num_heads*self.head_size, hidden_size)

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


class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, 4 * config.hidden_size)
        self.fc2 = Linear(4 * config.hidden_size, config.hidden_size)
        self.gelu = GELU()
        self.dropout = Dropout(config.dropout)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class DecoderBlock(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
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
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.dropout = Dropout(config.dropout)
        self.layers = Sequential(*[DecoderBlock(config) for _ in range(config.num_layers)])
        self.ln = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Function to print a dataframe visualising parameter count
        if config.print_param_count:
            input_data = t.randint(0, 1, (config.max_seq_len,), device=device)
            summary = torchinfo.summary(self, input_data=input_data)
            for line in str(summary).split("\n"): print(line)

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

config = TransformerConfig(
    num_layers = 2,
    num_heads = 6,
    vocab_size = trainset.vocab_size,
    hidden_size = 96,
    max_seq_len = trainset.seq_len,
    print_param_count=True
)

model = DecoderOnlyTransformer(config).to(device).train()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 1

# %%

model = train(model, optimizer, loss_fn, trainloader, epochs)

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
