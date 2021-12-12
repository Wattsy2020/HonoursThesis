# This uses code from the FEAT model, code taken from here https://github.com/Sha-Lab/FEAT/blob/master/model/models/feat.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import fewshot_re_kit

class MetaFormer(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, heads=1, euclidean=False, temperature=1, temperature2=1, dropout=0.5):
        super().__init__(sentence_encoder)
        hidden_size = sentence_encoder.config["hidden_size"]
        self.hidden_size = 2*hidden_size if sentence_encoder.cat_entity_rep else hidden_size # if the entity rep is concatenated the size will be doubled
        
        attn_d = int(self.hidden_size/heads)
        self.self_attn = MultiHeadAttention(heads, attn_d, attn_d, attn_d, dropout=dropout)     
        self.euclidean = euclidean
        self.temperature = temperature
        self.temperature2 = temperature2
        
    def forward(self, support, query, N, K, total_Q, is_student=False, regularisation=False):
        # get encoding vectors
        support = self.sentence_encoder(support)["encoding"]
        query = self.sentence_encoder(query)["encoding"]
        support = support.view(N, K, self.hidden_size) 
        query = query.view(total_Q, 1, self.hidden_size) 
        
        # adapt the support set
        support = support.view(N*K, self.hidden_size) # flatten so that we can pass to self attention
        support, attn = self.self_attn(support, support, support, N, K)
        prototypes = support.view(N, K, self.hidden_size).mean(dim=1) # for now simply average to get the prototype
        
        # calculate distance to get logits
        if self.euclidean: # use some expanding so that we can calculate the distance between each query and each prototype in one subtraction
            prototypes = prototypes.unsqueeze(1).expand(N, total_Q, self.hidden_size).contiguous()
            prototypes = prototypes.transpose(0, 1)
            query = query.expand(total_Q, N, self.hidden_size)

            logits = - torch.sum(torch.pow(prototypes - query, 2), 2) / self.temperature
        else:
            assert prototypes.shape[0] == N and prototypes.shape[1] == self.hidden_size
            prototypes = F.normalize(prototypes, dim=-1) # normalize for cosine similarity
            #query = F.normalize(query, dim=-1) 
            # not mentioned in the paper but they don't normalize both vectors when calculating cos sim https://github.com/Sha-Lab/FEAT/issues/62
            # in a later paper they highlight that this is equivalent to choosing a dynamic temperature = L2 norm of query, https://arxiv.org/pdf/2011.14663.pdf
            query = query.view(total_Q, self.hidden_size)
            
            logits = torch.matmul(query, prototypes.T) / self.temperature
        _, pred = torch.max(logits, dim=1)
        
        # Compute the centroids for each class and calculate a loss encouraging class samples to be closer to it 
        if regularisation:
            aux_task = torch.cat([support.view(K, N, self.hidden_size), query.view(K, N, self.hidden_size)], 0) # Interestingly they use the query samples for the auxillary task as well
            aux_task = aux_task.permute([1, 0, 2]).contiguous().view(N*2*K, self.hidden_size)
            
            # apply the transformation over the Aug Task and compute center
            aux_emb, _ = self.self_attn(aux_task, aux_task, aux_task, N, 2*K)
            aux_emb = aux_emb.view(N, 2*K, self.hidden_size)
            aux_center = torch.mean(aux_emb, 1).view(N, self.hidden_size)
            
            # Compute distance between the centers and the original embeddings (before passing through set-to-set function)
            # Interestingly this isn't the loss described in the paper, which described distance between the embeddings after transformation
            # This was also found by https://github.com/Sha-Lab/FEAT/issues/63
            if self.euclidean:
                raise ValueError("Euclidean distance doesn't work so this isn't implemented")
            else:
                # normalize for cosine similarity
                aux_center = F.normalize(aux_center, dim=-1)
                aux_task = aux_task.view(N, 2*K, self.hidden_size)
                aux_center = aux_center.T.unsqueeze(0).expand(N, self.hidden_size, N).contiguous() # create a batch dimension and duplicate the centers N times, so that torch.bmm works 
    
                # calculate distances between each K examples of N and the center
                logits_reg = torch.bmm(aux_task, aux_center) / self.temperature2
                logits_reg = logits_reg.view(N, 2*K, N)  
            
            # Define labels and calculate loss
            # the support and query set are constructed so that the first K elements belong to class 0, next K elements to class 1 etc... so we can define the label matrix here
            labels = torch.arange(N, dtype=torch.int64).repeat(2*K).reshape(2*K, N).T.cuda() # = concatentation of row vectors [i]*(2k) for i in [0, 9] 
            loss = F.cross_entropy(logits_reg, labels)
            return logits, pred, loss            
        elif is_student:
            return logits, pred, support, attn
        return logits, pred

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention with positional weighting'''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        
        # Create the attention weight parameters
        self.attn_weight = torch.nn.Parameter(torch.empty(3, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.attn_weight, gain=torch.nn.init.calculate_gain('tanh'))
        self.tanh = torch.nn.Tanh() # to scale the weights by a factor in [1, -1] i.e. similar samples from other classes should maybe be subtracted

    # create the attention weight matrix for an N-way K-shot problem
    def get_attn_weight(self, N, K):
        w_same_sample, w_same_class, w_diff_class = self.tanh(self.attn_weight)
        weights = torch.zeros((N*K, N*K)).cuda() + w_diff_class
        
        for row in range(len(weights)):
            weights[row, int(np.floor(row/K)*K):int(np.ceil((row+1)/K)*K)] = w_same_class
            weights[row, row] = w_same_sample
        return weights
        
    def forward(self, q, k, v, N, K):
        # compute attention
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        
        # weight it by creating a weight matrix from the 3 weights
        weights = self.get_attn_weight(N, K)
        attn = attn*weights
        
        # compute output
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module with residual sum built in'''
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, N, K):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        len_q, _ = q.size()
        len_k, _ = k.size()
        len_v, _ = v.size()

        # Pass through linear heads
        residual = q.clone()
        q = self.w_qs(q).view(len_q, n_head, d_k)
        k = self.w_ks(k).view(len_k, n_head, d_k)
        v = self.w_vs(v).view(len_v, n_head, d_v)
        
        # Resize to make the head dimension first, the calculate attention
        q = q.permute(1, 0, 2).contiguous() # n x lq x dk
        k = k.permute(1, 0, 2).contiguous() # n x lk x dk
        v = v.permute(1, 0, 2).contiguous() # n x lv x dv
        output, attn = self.attention(q, k, v, N, K)

        # Resize to (sequence_length, dimension)
        output = output.view(n_head, len_q, d_v)
        output = output.permute(1, 0, 2).contiguous().view(len_q, -1) # lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn
