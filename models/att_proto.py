# Defines a modified version of prototypical networks that also incorporates attention between query and support set to calculate prototypes
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

import fewshot_re_kit

def euclid_distance(a, b):
    return torch.sqrt(torch.sum(torch.pow(a - b, 2), dim=-1))

class att_proto(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, dropout=0.5):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        hidden_size = sentence_encoder.config["hidden_size"]
        self.hidden_size = 2*hidden_size if sentence_encoder.cat_entity_rep else hidden_size
        self.scale_factor = torch.sqrt(torch.tensor(self.hidden_size).float())
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)
        
        # for residual connection
        self.fc = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # heads for attention
        self.w_q = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_k = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_v = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
    def forward(self, support, query, N, K, total_Q, surrogate_support=None, surrogate_query=None, is_student=False, regularisation=False, visualisation=False):
        support = self.sentence_encoder(support)["encoding"]
        query = self.sentence_encoder(query)["encoding"]
        support = support.view(N, K, self.hidden_size)
        query = query.view(total_Q, 1, self.hidden_size)
        
        # Calculate Prototypes
        residual_proto = torch.mean(support, dim=1).unsqueeze(0).expand(total_Q, N, self.hidden_size) # calculate the standard prototype as a residual
        attQ = self.w_q(query) # apply linear layers to query, keys and vectors
        attK = self.w_k(support)
        attV = self.w_v(support)
        
        # calculate query weighted attention with each classes support set
        att = torch.matmul(attQ.view(total_Q, self.hidden_size), attK.view(N*K, self.hidden_size).T)/self.scale_factor # matrix of size (total_Q, N*K)
        att = att.view(total_Q, N, K)
        att = self.softmax(att) # softmax across the K examples of each class
        query_proto = torch.bmm(att.reshape(total_Q*N, 1, K), attV.unsqueeze(0).expand(total_Q, N, K, self.hidden_size).reshape(total_Q*N, K, self.hidden_size))
        query_proto = query_proto.reshape(total_Q, N, self.hidden_size)
        
        # finally apply a linear layer to the output, then add the original class mean as a sort of "residual"
        prototypes = self.layer_norm(self.dropout(self.fc(query_proto)) + residual_proto)
        
        # Calculate Logits
        query = query.expand(total_Q, N, self.hidden_size)
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