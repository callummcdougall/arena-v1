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
from typing import Optional

from functions_from_previous_days import *
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
    dropout: float
    layer_norm_epsilon: float
    print_param_count: bool = True

config = TransformerConfig(
    num_layers = 12,
    num_heads = 12,
    vocab_size = 28996,
    hidden_size = 768,
    max_seq_len = 512,
    dropout = 0.1,
    layer_norm_epsilon = 1e-12,
    print_param_count = False
)

# %%
# ============================= TRANSFORMER ARCHITECTURE =============================

class MultiheadAttention(nn.Module):
    W_Q: nn.Linear
    W_K: nn.Linear
    W_V: nn.Linear
    W_O: nn.Linear

    def __init__(self, config: TransformerConfig):
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
        self.W_Q = nn.Linear(self.hidden_size, self.num_heads*self.head_size)
        self.W_K = nn.Linear(self.hidden_size, self.num_heads*self.head_size)
        self.W_V = nn.Linear(self.hidden_size, self.num_heads*self.head_size)
        self.W_O = nn.Linear(self.num_heads*self.head_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor]) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        attention_values = self.multihead_attention(Q, K, V, additive_attention_mask, num_heads=self.num_heads)

        output = self.W_O(attention_values)

        return self.dropout(output)

    # Now moving this function into a class method, so it can refer to the dropout layers
    def multihead_attention(self, Q: t.Tensor, K: t.Tensor, V: t.Tensor, additive_attention_mask: Optional[t.Tensor], num_heads: int) -> t.Tensor:

        q = rearrange(Q, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
        k = rearrange(K, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
        v = rearrange(V, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)

        batch, seq_len, nheads, headsize = q.shape
        attention_scores = einsum("batch seqQ nheads headsize, batch seqK nheads headsize -> batch nheads seqQ seqK", q, k) / (headsize ** 0.5)
        if additive_attention_mask is not None:
            attention_scores = attention_scores + additive_attention_mask

        attention_probabilities = attention_scores.softmax(dim=-1)

        attention_values = einsum("batch nheads seqQ seqK, batch seqK nheads headsize -> batch seqQ nheads headsize", attention_probabilities, v)

        out = rearrange(attention_values, "batch seqQ nheads headsize -> batch seqQ (nheads headsize)")

        return out


class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class BERTBlock(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = MultiheadAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        x = self.ln1(x + self.attn(x, additive_attention_mask))
        x = self.ln2(x + self.mlp(x))
        return x


def make_additive_attention_mask(one_zero_attention_mask: t.Tensor, big_negative_number: float = -10000) -> t.Tensor:
    '''
    one_zero_attention_mask: 
        shape (batch, seq)
        Contains 1 if this is a valid token and 0 if it is a padding token.

    big_negative_number:
        Any negative number large enough in magnitude that exp(big_negative_number) is 0.0 for the floating point precision used.

    Out: shape (batch, heads, seq, seq). Contains 0 if attention is allowed, and big_negative_number if it is not allowed.
    '''
    return big_negative_number * repeat(1 - one_zero_attention_mask, "batch seqK -> batch 1 1 seqK")

        
class BertCommon(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_encoding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.token_type_embedding = nn.Embedding(2, config.hidden_size)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([BERTBlock(config) for _ in range(config.num_layers)])



        # Function to print a Dframe visualising parameter count (this can be omitted, but it's pretty useful!)
        if config.print_param_count:
            utils.print_param_count(self)
            
    def forward(
        self,
        input_ids: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        """
        input_ids: (batch, seq) - the token ids
        token_type_ids: (batch, seq) - only used for next sentence prediction.
        one_zero_attention_mask: (batch, seq) - only used in training. See make_additive_attention_mask.
        """
        if one_zero_attention_mask is None:
            additive_attention_mask = None
        else:
            additive_attention_mask = make_additive_attention_mask(one_zero_attention_mask)
        
        token_emb = self.token_embedding(input_ids)

        posn = t.arange(input_ids.shape[1], device=input_ids.device)
        pos_emb = self.positional_encoding(posn)
        
        if token_type_ids is None:
            token_type_ids = t.zeros_like(input_ids, dtype=t.int64)
        tokentype_emb = self.token_type_embedding(token_type_ids)
        
        x = token_emb + pos_emb + tokentype_emb

        x = self.ln1(x)
        x = self.dropout(x)

        for bertblock in self.layers:
            x = bertblock(x, additive_attention_mask)
        
        return x
    

class BertLanguageModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bertcommon = BertCommon(config)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.unembed_bias = nn.Parameter(t.zeros(config.vocab_size))

    def forward(
        self,
        input_ids: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        """Compute logits for each token in the vocabulary.

        Return: shape (batch, seq, vocab_size)
        """
        x = self.bertcommon(input_ids, one_zero_attention_mask, token_type_ids)

        x = self.ln2(self.gelu(self.fc(x)))
        x = einsum("batch seq hidden, vocab hidden -> batch seq vocab", x, self.bertcommon.token_embedding.weight) + self.unembed_bias
        return x

# %%
# ============================= INITIALISING MODELS =============================

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

my_bert = BertLanguageModel(config).train()
bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")

utils.print_param_count(my_bert, bert)

# %%
# ============================= LOADING WEIGHTS =============================

def copy_weights_from_bert(my_bert: BertLanguageModel, bert: transformers.models.bert.modeling_bert.BertForMaskedLM) -> BertLanguageModel:
    '''
    Copy over the weights of `bert` to your bert implementation.
    
    This is a bit messier than GPT, because the layers don't naturally line up (you can see this from running `utils.print_param_count`).

    The difference comes from `unembed_bias`, which gets put at the top of `my_bert.parameters()` even though it's near the bottom in HuggingFace's model.

    Solution is to use a list rather than a dict, then I can reorder it so that the params do have the right 1-1 correspondence.

    Another difference than the other copy weights function: by default, these linear layer weights are equal, not equal transposed! Unlike GPT.
    '''

    my_bert_list = list(my_bert.named_parameters())
    bert_list = list(bert.named_parameters())
    # This is the reordering step
    bert_list = [bert_list[-5]] + bert_list[:-5] + bert_list[-4:]
    
    # Initialise an empty dictionary to store the correct key-value pairs
    state_dict = {}

    # Check the number of params/buffers is correct
    assert len(my_bert_list) == len(bert_list), "Number of layers is wrong."
    
    for (my_param_name, my_param), (name, param) in zip(my_bert_list, bert_list):
        state_dict[my_param_name] = param

    if set(state_dict.keys()) != set(my_bert.state_dict().keys()):
        raise Exception("State dicts don't match.")
    
    my_bert.load_state_dict(state_dict)
    
    return my_bert

my_bert = copy_weights_from_bert(my_bert, bert)


# %%
# ============================= MAKING PREDICTIONS =============================


def predict(model, tokenizer, text: str, k=15) -> List[List[str]]:
    """
    Return a list of k strings for each [MASK] in the input.
    """
    model.eval()

    # Get input ids, and generate output
    input_ids = tokenizer.encode(text=text, return_tensors="pt")
    output_logits = model(input_ids)
    # This deals with the case where the model I'm using is HuggingFace
    if not isinstance(output_logits, t.Tensor):
        output_logits = output_logits.logits
    
    # Iterate through the input_ids, and add predictions for each masked token
    mask_predictions = []
    for i, input_id in enumerate(input_ids.squeeze()):
        if input_id == tokenizer.mask_token_id:
            logits = output_logits[0, i]
            top_logits_indices = t.topk(logits, k).indices
            predictions = tokenizer.decode(top_logits_indices)
            mask_predictions.append(predictions)
    
    return mask_predictions

def test_bert_prediction(predict, model, tokenizer):
    """Your Bert should know some names of American presidents."""
    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text)
    print(f"Prompt: {text}")
    print("Model predicted: \n", "\n".join(map(str, predictions)), "\n")
    assert "Washington" in predictions[0]
    assert "Bush" in predictions[0]

test_bert_prediction(predict, my_bert, tokenizer)

your_text = "The Answer to the Ultimate Question of Life, The Universe, and Everything is [MASK]."
predictions = predict(my_bert, tokenizer, your_text)
print("Model predicted: \n", "\n".join(map(str, predictions)))

# %%
