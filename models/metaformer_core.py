# A greatly simplified version of metaformer. The only operation it performs as the attention weighted sum
# There are no linear layers before the attention, the attention weights are the only learnt parameters
# This is to study how the attention weights are learnt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import fewshot_re_kit

class MetaFormerCore(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, nlayers=1, nheads=1, dropout=0):
        super().__init__(sentence_encoder)
        hidden_size = sentence_encoder.config["hidden_size"]
        self.hidden_size = 2*hidden_size if sentence_encoder.cat_entity_rep else hidden_size # if the entity rep is concatenated the size will be doubled
        
        attn_d = int(self.hidden_size/nheads)
        self.encoder = Encoder(nlayers, nheads, self.hidden_size, attn_d, attn_d, dropout=dropout)
        
    def forward(self, support, query, N, K, total_Q, is_student=False, visualisation=False):
        # get encoding vectors
        support = self.sentence_encoder(support)["encoding"]
        query = self.sentence_encoder(query)["encoding"]
        support = support.view(N, K, self.hidden_size) 
        query = query.view(total_Q, 1, self.hidden_size)
        
        # adapt the samples
        # First concatenate the support and query set together, we want to add each query individually so this is an inductive algorithm
        support_adapt = support.view(N*K, self.hidden_size).unsqueeze(0).expand(total_Q, N*K, self.hidden_size)
        original_samples = support_adapt
        samples = torch.cat([support_adapt, query], dim=1) # add 1 query sample to each batch
        samples, attn, pre_residual = self.encoder(samples, N, K) # samples has shape 
        samples = samples.view(total_Q, N*K+1, self.hidden_size) # the attention operation preserves the shape
        
        # Get different prototypes for each batch and calculate distance between them and the query
        #prototypes = samples[:, :-1, :].view(total_Q, N, K, self.hidden_size).mean(dim=2) # for now simply average to get the prototype
        prototypes = samples[:, :-1, :].view(total_Q, N, K, self.hidden_size)[:, :, 0, :] # take the first example, removing averaging
        
        # calculate distance to get logits
        """ cosine distance
        prototypes = F.normalize(prototypes, dim=-1) # normalize for cosine similarity
        query_adapt = samples[:, -1, :].view(total_Q, 1, self.hidden_size) # keep dim 1 for batch matrix multiply
        logits = torch.bmm(query_adapt, prototypes.transpose(1, 2)).view(total_Q, N)
        """
        query_adapt = samples[:, -1, :].view(total_Q, 1, self.hidden_size).expand(total_Q, N, self.hidden_size)
        distance = torch.sqrt(torch.sum(torch.pow(prototypes - query_adapt, 2), dim=-1))
        logits = -distance.view(total_Q, N)
        _, pred = torch.max(logits, dim=1)
        
        if is_student:
            return logits, pred, support, attn
        elif visualisation:
            return logits, pred, original_samples, pre_residual, samples
        return logits, pred

    # loop through all heads, get all attention weights, and stack them into one tensor
    def get_attn_weight_matrix(self, N, K, layer, bias=False):
        weights = [head.get_attn_weight(N, K, bias=bias) for head in self.encoder.attn_layers[layer].attn_heads]
        weights = torch.stack(weights, dim=0)
        return weights[0] # until train_demos call and plotting of this is changed
    
    # return the raw attn_weight parameters for each head
    def get_attn_weights(self, layer=0):
        weights = [head.attn_weight.T for head in self.encoder.attn_layers[layer].attn_heads]
        weights = torch.cat(weights, dim=0) # dimension of (nhead, 6)
        return weights

class Encoder(nn.Module):
    ''' Encapsulates multiple layers of MultiHeadAttention and Dense Feed forward networks'''
    def __init__(self, nlayers, nhead, d_model, d_k, d_v, dropout=0):
        super().__init__()
        
        # Attention layers
        self.nlayers = nlayers
        attn_layers = [MultiHeadAttention(nhead, d_model, d_k, d_v, dropout=dropout) for _ in range(nlayers)]
        self.attn_layers = torch.nn.ModuleList(attn_layers) # properly registers the list of modules, so that .cuda() converts them all to cuda
    
    def forward(self, samples, N, K, query=True):
        attentions = []
        for i in range(self.nlayers):
            samples, attn, pre_residual = self.attn_layers[i](samples, samples, samples, N, K, query=query)
        return samples, attentions, pre_residual

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module with residual sum built in'''
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        # Multiple attention heads, each with different learnt attention weighs
        # setting temperature = np.sqrt(d_k) enables scaled dot product attention
        self.attn_heads = torch.nn.ModuleList([ScaledDotProductAttention(temperature=np.sqrt(d_k), dropout=dropout) for _ in range(self.n_head)])
        
        # for the residual connection, add this in later as it is an important part of MHA
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, N, K, query=True):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q.clone()
        
        # resize
        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)
        
        # transpose to make the head dimension first, then calculate attention for each head
        q = q.permute(2, 0, 1, 3).contiguous()
        k = k.permute(2, 0, 1, 3).contiguous()
        v = v.permute(2, 0, 1, 3).contiguous()
        head_outputs = []
        for i in range(n_head):
            output, attn = self.attn_heads[i](q[i], k[i], v[i], N, K, query=query)
            head_outputs.append(output)
        output = torch.cat(head_outputs, dim=-1).view(sz_b, len_q, self.d_model) # concat and resize to (batch, sequence_length, dimension)
        attn = attn.mean(dim=0) # take the mean of the attentions over the queries so we can plot them, TODO: update to work with multiple attention heads

        # dropout and add residual
        pre_residual = output
        #output = self.layer_norm(self.dropout(output) + residual)
        return output, attn, pre_residual

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention with positional weighting'''
    def __init__(self, temperature, dropout=0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        
        # Create the attention weight parameters
        self.attn_weight = torch.nn.Parameter(torch.empty(6, 1), requires_grad=True)
        self.attn_bias = torch.nn.Parameter(torch.zeros(6, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.attn_weight, gain=torch.nn.init.calculate_gain('tanh')) # it is hard to learn attn_weight if they're initialized to the same values
        #self.attn_weight = torch.nn.Parameter(torch.tensor([1, 1, 0, 0, 0, 1]).double(), requires_grad=True) # initialise to prototypical networks
        #torch.nn.init.xavier_uniform_(self.attn_bias, gain=torch.nn.init.calculate_gain('tanh'))

    # create the attention weight matrix for an N-way K-shot problem
    def get_attn_weight(self, N, K, bias=False):
        if bias:
            w_same_sample, w_same_class, w_diff_class, w_stoquery, w_querytos, w_query = self.attn_bias
        else:
            w_same_sample, w_same_class, w_diff_class, w_stoquery, w_querytos, w_query = self.attn_weight
        weights = torch.zeros((N*K + 1, N*K + 1)).cuda()
        
        # Set the weights for the support set
        for row in range(N*K):
            weights[row, :] = w_diff_class
            weights[row, -1] = w_stoquery
            weights[row, int(np.floor(row/K)*K):int(np.ceil((row+1)/K)*K)] = w_same_class
            weights[row, row] = w_same_sample
            
        # Set the weights for the query, it must always be the last item in the sequence 
        weights[-1, :-1] = w_querytos
        weights[-1, -1] = w_query
        return weights
        
    def forward(self, q, k, v, N, K, query=True): 
        # compute attention
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn) # has size Total_Q, N*K+1, N*K+1
        
        # weight it by using a weight matrix constructed from the learnt weights
        weights = self.get_attn_weight(N, K)
        bias = self.get_attn_weight(N, K, bias=True)
        if not query: # during regularisation no query is provided, in which case we simply trim the weight matrix to remove the query related weights
            weights = weights[:-1, :-1]
            bias = bias[:-1, :-1]
        attn = torch.tanh(weights) * attn # best performer so far
        #attn = torch.tanh(weights*attn + bias) # second best performer
        #attn = torch.tanh(weights) * attn + torch.tanh(bias) # mostly just worse
        #attn = attn + torch.tanh(bias) # little change from prototypical networks
        
        # compute output
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn
