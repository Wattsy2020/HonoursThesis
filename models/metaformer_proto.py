# Defines a modified version of prototypical networks that includes the key ideas such as attention and attention weights 
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

import fewshot_re_kit

def euclid_distance(a, b):
    return torch.sqrt(torch.sum(torch.pow(a - b, 2), dim=-1))

class MetaFormerProto(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, dropout=0.5, att_weight_reg=False):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        hidden_size = sentence_encoder.config["hidden_size"]
        self.hidden_size = 2*hidden_size if sentence_encoder.cat_entity_rep else hidden_size
        self.scale_factor = torch.sqrt(torch.tensor(self.hidden_size).float())
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)
        self.attn_matrix = WeightedAttentionMatrix(att_weight_reg = att_weight_reg)
        
    def forward(self, support, query, N, K, total_Q, surrogate_support=None, surrogate_query=None, is_student=False, regularisation=False, visualisation=False):
        support = self.sentence_encoder(support)["encoding"]
        query = self.sentence_encoder(query)["encoding"]
        support = self.dropout(support.view(N, K, self.hidden_size))
        query = self.dropout(query.view(total_Q, 1, self.hidden_size))
        
        # Calculate Prototypes
        prototypes = torch.mean(support, dim=1)
        prototypes = prototypes.unsqueeze(0).expand(total_Q, N, self.hidden_size)
        
        # Perform attention weighted sum to mix query and support together
        episode = torch.cat([prototypes, query], dim=1).view(total_Q, N+1, self.hidden_size)
        weight_matrix = self.attn_matrix.get_matrix(N, 1).unsqueeze(0).expand(total_Q, N+1, N+1)
        bias_matrix = self.attn_matrix.get_matrix(N, 1, bias=True).unsqueeze(0).expand(total_Q, N+1, N+1)
        
        attn = torch.bmm(episode, episode.transpose(1, 2))/self.scale_factor
        attn = weight_matrix*(self.softmax(attn) + bias_matrix)
        attn = self.dropout(attn)
        episode = torch.bmm(attn, episode)
        
        # Calculate Logits
        prototypes = episode[:, :-1, :]
        query = episode[:, -1, :].view(total_Q, 1, self.hidden_size).expand(total_Q, N, self.hidden_size)
        logits = -euclid_distance(prototypes, query).view(total_Q, N)
        _, pred = torch.max(logits, dim=1)
        return logits, pred
        
    # loop through all heads, get all attention weights, and stack them into one tensor
    def get_attn_weight_matrix(self, N, K, layer):
        weights = self.attn_matrix.get_matrix(N, K)
        return weights
    
    # return the raw attn_weight parameters for each head
    def get_attn_weights(self, layer=0):
        weights = torch.cat([self.attn_matrix.attn_weight, self.attn_matrix.attn_bias], dim=1).T
        return weights
    
class WeightedAttentionMatrix(nn.Module):
    ''' Creates the attention weight matrix '''
    def __init__(self, att_weight_reg=False):
        super().__init__()
        self.att_weight_reg = att_weight_reg
        
        if self.att_weight_reg:
            self.attn_weight = torch.nn.Parameter((torch.tensor([1, 1, 1, 1, 1, 1]).float()).reshape(6, 1), requires_grad=True)
        else: # initialise near the default prototypical networks
            self.attn_weight = torch.nn.Parameter((torch.tensor([1, 1, -0.2, 0.2, 0.2, 1]).float()).reshape(6, 1), requires_grad=True)
        
        # Other initialisation methods
        # self.attn_weight = torch.nn.Parameter(torch.tensor([1, 0.5, -0.5, 0.5, 0.2, 1]), requires_grad=True) # hypothetical optimal initialisation

        # Optionally can use a bias as an addition or the sole weight
        self.attn_bias = torch.nn.Parameter(torch.zeros(6, 1), requires_grad=True)

    # create the attention weight matrix for an N-way K-shot problem
    def get_matrix(self, N, K, bias=False, query=True):
        if not bias:
            weight_vector = self.attn_weight
        else:
            weight_vector = self.attn_bias
        w_same_sample, w_same_class, w_diff_class, w_stoquery, w_querytos, w_query = weight_vector
        weights = torch.zeros((N*K + 1, N*K + 1)).cuda()
        
        # Set the weights for the support set
        for row in range(N*K):
            weights[row, :] = w_diff_class
            weights[row, -1] = w_stoquery
            weights[row, int(np.floor(row/K)*K):int(np.ceil((row+1)/K)*K)] = w_same_class
            weights[row, row] = w_same_sample
            
        # Set the weights for the query, it must always be the last item in the sequence
        if query:
            weights[-1, :-1] = w_querytos
            weights[-1, -1] = w_query
        else: # remove query weights
            weights = weights[:-1, :-1]
        return weights
