# A reimplementation of nn.transformers that outputs the intermediate hidden states and attention matrices
# (original source here: https://pytorch.org/docs/1.7.0/_modules/torch/nn/modules/transformer.html)
import copy
from typing import Optional, Any, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm

# We need an encoder only bidirectional transformer like BERT, so we remove the decoder functionality
class Transformer(Module):
    def __init__(self, d_model: int = 624, nhead: int = 8, num_encoder_layers: int = 6, dim_feedforward: int = 624, 
                 dropout: float = 0.5, activation: str = "gelu", attn_style: str = "standard"):
        super(Transformer, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, attn_style=attn_style)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self.d_model = d_model
        self.nhead = nhead

    # Takes in embeddings, passes it through the transformer and returns the output at the final layer (the MetaFormer computs output logits using this encoding)
    def forward(self, embeds: Tensor, need_hidden: bool = False, need_weights: bool = False) -> Tuple[Tensor]:
        embeds *= np.sqrt(self.d_model)  # to scale up the embeddings, so the positional embedding doesn't overwrite their meaning
        embeds = self.pos_encoder(embeds)
        output = self.encoder(embeds, need_hidden=need_hidden, need_weights=need_weights)
        return output

# This is based on examples provided by pytorch https://github.com/pytorch/examples/tree/master/word_language_model
# It reimplements the sinusoidal positional embedding in Attention is all you need
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
                

class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, need_hidden: bool = False, need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            need_hidden: if true return all the hidden states as a list of tensors
            need_weights: if true return all the attention weights as a list of tensor
        """
        output = src
        hidden_states = []
        att_weights = []
        
        for mod in self.layers:
            output = mod(output, need_weights=need_weights)
            if need_weights:
                att_weights.append(output[1])
                output = output[0]
            if need_hidden:
                hidden_states.append(output)

        if self.norm is not None:
            output = self.norm(output)

        output = [output]
        if need_hidden:
            output.append(hidden_states)
        if need_weights:
            output.append(att_weights)
        if len(output) == 1:
            output = output[0]
        return output

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need"

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        attn_style: either "standard" or "learnt" in which case the attention matrix is multiplied by learnt weights pre-softmax 

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", attn_style="standard", seq_length=73):
        super(TransformerEncoderLayer, self).__init__()
        if attn_style=="standard":
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        elif attn_style=="learnt":
            self.self_attn = LearntAttention(d_model, seq_length, nhead=nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            need_weights: whether or not to return the attention weights
        """
        # note that the att_weights are averaged across the heads according to https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py multi_head_attention_forward
        src2, att_weights = self.self_attn(src, src, src, need_weights=True) 
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        if need_weights:
            return src, att_weights
        return src
    
    
# An alternate version of attention where the attention matrices are fixed and learnt by gradient descent
# helpful is the source code for pytorch's multiheadattention: https://pytorch.org/docs/1.7.0/_modules/torch/nn/modules/activation.html#MultiheadAttention    
class LearntAttention(torch.nn.Module):
    def __init__(self, d_model, seq_length, nhead=1, dropout=0):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        
        # Define linear heads, TODO: add multi head attention
        self.query_proj = torch.nn.Linear(d_model, d_model)
        self.key_proj = torch.nn.Linear(d_model, d_model)
        self.value_proj = torch.nn.Linear(d_model, d_model)
        self.softmax = torch.nn.Softmax(dim=-1)
    
        # Define the learnable attention weighting matrix
        standard_init = True # if true use xavier
        if standard_init:
            self.attn_weight = torch.nn.Parameter(torch.empty(seq_length, seq_length), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.attn_weight, gain=torch.nn.init.calculate_gain('tanh')) # standard initialisation
        else: # use the ideal attention matrix of 1 if same class else 0
            ideal_att = torch.zeros(seq_length, seq_length).cuda()
            K = 5 # TODO: unhardcode this
            for row in range(seq_length - 3): # -3 ignores the query and markers
                if row % (K+2) == 0 or (row+1) % (K+2) == 0: # skip the marker rows
                    continue 
                start_class = int(np.floor(row/(K+2))*(K+2) + 1) # position of the first example of this class
                end_class = int(np.ceil(row/(K+2))*(K+2) - 2)    # position of last example of this class
                ideal_att[row, start_class:(end_class+1)] = 1
            self.attn_weight = torch.nn.Parameter(torch.clone(ideal_att), requires_grad=True)
            
        self.tanh = torch.nn.Tanh() # to scale the weights by a factor in [1, -1] i.e. similar samples from other classes should maybe be subtracted
    
    def forward(self, query, key, value, need_weights=False):
        # if the sequence dimension is first swap it with the batch dimension
        transpose = query.shape[0] == self.seq_length and query.shape[1] != self.seq_length # avoid edge case where batch_size==seq_length
        if transpose: 
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # apply corresponding linear heads to query, key, value 
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        
        # then calculate attention, this link is useful: https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/nn/functional.py#L4811
        # TODO consider dropout on the attention here as in the linked function
        query = query / np.sqrt(self.d_model) # scale the attention
        attn = torch.bmm(query, key.transpose(-2, -1)) # (B, seq_length, d) * (B, d, seq_length) = (B, seq_length, seq_length)
        attn = self.softmax(attn)
        attn = self.attn_weight*attn # weight the attention
        output = torch.bmm(attn, value)
        
        if transpose: # return to original shape
            output = output.transpose(0, 1)
        if need_weights:
            return output, attn
        return output
    
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
