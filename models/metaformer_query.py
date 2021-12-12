# This uses code from the FEAT model, code taken from here https://github.com/Sha-Lab/FEAT/blob/master/model/models/feat.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

import fewshot_re_kit
import models.regularization as regularization

def euclid_distance(a, b):
    return torch.sqrt(torch.sum(torch.pow(a - b, 2), dim=-1))

class MetaFormerQuery(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, nlayers=1, nheads=1, euclidean=False, temperature=1, temperature2=1, dropout=0.5, combined=False,
                dense=False, surrogate_novel=False, att_type="scaled_dot", ablate_learnt_att=False, ablate_att=False, 
                ablate_mpe_reg = False, ablate_feat_reg = False, att_weight_reg=False, 
                pre_avg=False, ablate_query=False, att_bias=False):
        super().__init__(sentence_encoder)
        hidden_size = sentence_encoder.config["hidden_size"]
        self.hidden_size = 2*hidden_size if sentence_encoder.cat_entity_rep else hidden_size # if the entity rep is concatenated the size will be doubled
        self.max_batch_size = 300
        
        self.euclidean = euclidean
        self.temperature = temperature
        self.temperature2 = temperature2
        self.surrogate_novel = surrogate_novel
        
        self.att_weight_reg = att_weight_reg
        self.weight_loss = torch.nn.MSELoss()
        #self.weight_loss = torch.nn.L1Loss()
        
        self.pre_avg = pre_avg # whether to average the support set before or after the encoding
        self.ablate_query = ablate_query # remove the query from input to the encoder, only comparing with it at the end
        self.combined = combined # combine both pre and post averaging by passing the average vectors in as part of the support set
        self.ablate_mpe_reg = ablate_mpe_reg
        self.ablate_feat_reg = ablate_feat_reg
        
        attn_d = int(self.hidden_size/nheads)
        self.att_bias = att_bias
        self.encoder = Encoder(nlayers, nheads, self.hidden_size, attn_d, attn_d, dropout=dropout, dense=dense, 
                               att_type=att_type, ablate_learnt_att=ablate_learnt_att, ablate_att=ablate_att, att_weight_reg=att_weight_reg,
                               att_bias=att_bias)
        
    def forward(self, support, query, N, K, total_Q, surrogate_support=None, surrogate_query=None, is_student=False, regularisation=False, visualisation=False):
        # get encoding vectors
        support = self.sentence_encoder(support)["encoding"]
        query = self.sentence_encoder(query)["encoding"]
        support = support.view(N, K, self.hidden_size) 
        query = query.view(total_Q, 1, self.hidden_size)
        
        # Add in the surrogate query and support set
        if surrogate_support is not None and surrogate_query is not None:
            support = torch.cat([support, surrogate_support], axis=0)
            surrogate_query = surrogate_query.reshape(-1, 1, self.hidden_size) # flatten the query samples that originally have shape (N, K, hidden_size)
            query = torch.cat([query, surrogate_query], axis=0)
            
            # Update the shape parameters to avoid errors when reshaping
            N += surrogate_support.shape[0]
            total_Q += surrogate_query.shape[0]
        
        # adapt the samples
        # First concatenate the support and query set together, we want to add each query individually so this is an inductive algorithm
        support_adapt = support.view(N*K, self.hidden_size).unsqueeze(0).expand(total_Q, N*K, self.hidden_size)
        original_samples = support_adapt

        # Get different prototypes for each query
        if self.pre_avg:
            # take the mean of the support set to get prototypes, then adapt them
            samples = torch.cat([support_adapt.reshape(total_Q, N, K, -1).mean(dim=2), query], dim=1)
            samples, attn, pre_residual = self.encoder(samples, N, 1)
            prototypes = samples[:, :-1, :] # remove query to get the prototypes
        elif self.ablate_query:
            samples = support.view(N*K, self.hidden_size).unsqueeze(0).expand(1, N*K, self.hidden_size)
            samples, attn, pre_residual = self.encoder(samples, N, K, query=False)
            samples = samples.expand(total_Q, N*K, self.hidden_size)
            samples = torch.cat([samples, query], dim=1) # append the query at the end of samples so the remaining code works without modification
            prototypes = samples[:, :-1, :].view(total_Q, N, K, self.hidden_size).mean(dim=2) 
        elif self.combined:
            # get the pre averaged prototypes, then add them into the standard support set
            support_adapt = support_adapt.reshape(total_Q, N, K, -1)
            pre_avg_proto = support_adapt.mean(dim=2).unsqueeze(2).reshape(total_Q, N, 1, self.hidden_size)
            samples = torch.cat([pre_avg_proto, support_adapt], dim=2).reshape(total_Q, N*(K+1), self.hidden_size)
            samples = torch.cat([samples, query], dim=1) # add the query back in
            
            # Perform adaptation, get the final prototype from the adapted version of pre_avg_proto
            if samples.shape[1] < self.max_batch_size:
                samples, attn, pre_residual = self.encoder(samples, N, K+1)
            else:  # in validation the batch size is too large, it hits 354 which due to the O(n^3) complexity of matrix multiplications greatly increases memory needed
                batch_size = math.floor(samples.shape[0]/2) # we only need two batches
                samples1, attn1, pre_residual1 = self.encoder(samples[:batch_size, :, :], N, K+1)
                samples2, attn2, pre_residual2 = self.encoder(samples[batch_size:, :, :], N, K+1)
                samples, pre_residual = torch.cat([samples1, samples2], dim=0), torch.cat([pre_residual1, pre_residual2], dim=0)
                attn = attn1 # Note that attention is a list of the attentions for each layer, it should be fine to return attentions for half the prediction
            prototypes = samples[:, :-1, :].reshape(total_Q, N, K+1, self.hidden_size)[:, :, 0, :]
        else:
            samples = torch.cat([support_adapt, query], dim=1) # add 1 query sample to each batch
            samples, attn, pre_residual = self.encoder(samples, N, K) # samples has shape 
            samples = samples.view(total_Q, N*K+1, self.hidden_size) # the attention operation preserves the shape
            prototypes = samples[:, :-1, :].view(total_Q, N, K, self.hidden_size).mean(dim=2)
        
        # calculate distance to get logits
        if self.euclidean: # use some expanding so that we can calculate the distance between each query and each prototype in one subtraction
            query_adapt = samples[:, -1, :].view(total_Q, 1, self.hidden_size).expand(total_Q, N, self.hidden_size)
            distance = euclid_distance(prototypes, query_adapt)
            logits = -distance.view(total_Q, N)/ self.temperature
        else:
            prototypes = F.normalize(prototypes, dim=-1) # normalize for cosine similarity
            #query = F.normalize(query, dim=-1) 
            # not mentioned in the paper but they don't normalize both vectors when calculating cos sim https://github.com/Sha-Lab/FEAT/issues/62
            # in a later paper they highlight that this is equivalent to choosing a dynamic temperature = L2 norm of query, https://arxiv.org/pdf/2011.14663.pdf
            query_adapt = samples[:, -1, :].view(total_Q, 1, self.hidden_size) # keep dim 1 for batch matrix multiply
            
            logits = torch.bmm(query_adapt, prototypes.transpose(1, 2)) / self.temperature
            assert logits.shape[0] == total_Q and logits.shape[1] == 1 and logits.shape[2] == N, "logit shape incorrect"
            logits = logits.view(total_Q, N)
        _, pred = torch.max(logits, dim=1)
        
        # Compute the centroids for each class and calculate a loss encouraging class samples to be closer to it 
        if regularisation:
            # Construct the reglularisation task, a separate task from inference where both the support and query are adapted together
            aux_task = torch.cat([support.view(N, K, self.hidden_size), query.view(N, K, self.hidden_size)], 1)
            
            # apply the transformation, then compute centers using only the support set to match the classification task
            aux_emb, _, _ = self.encoder(aux_task.view(1, N*2*K, self.hidden_size), N, 2*K, query=False)
            aux_emb = aux_emb.view(N, 2*K, self.hidden_size)
            aux_center = torch.mean(aux_emb[:, :K, :], 1).view(N, self.hidden_size)
            
            mpe_reg = regularization.mpe_intra_inter_regularization(aux_center, aux_emb, N, 2*K) 
            feat_reg = regularization.mean_loss(aux_center, aux_task, N, 2*K)
            # or use cosine_intra_inter_regularization(aux_center, aux_emb, N, 2*K)
            
            loss = 0
            if not self.ablate_mpe_reg:
                loss += mpe_reg
            if not self.ablate_feat_reg:
                loss += 10*feat_reg
            if self.att_weight_reg:
                default_att_weight = torch.tensor([1, 1, 1, 1, 1, 1]).float().cuda()
                att_weight = self.get_attn_weights()[0].reshape(6)
                loss += 0.1*self.weight_loss(att_weight, default_att_weight)
            return logits, pred, loss            
        elif is_student:
            return logits, pred, support, attn
        elif visualisation:
            return logits, pred, original_samples, pre_residual, samples
        return logits, pred

    # loop through all heads, get all attention weights, and stack them into one tensor
    def get_attn_weight_matrix(self, N, K, layer):
        weights = [head.get_attn_weight(N, K) for head in self.encoder.attn_layers[layer].attn_heads]
        weights = torch.stack(weights, dim=0)
        return weights[0] # until train_demos call and plotting of this is changed
    
    # return the raw attn_weight parameters for each head
    def get_attn_weights(self, layer=0):
        weights = [head.attn_weight.T for head in self.encoder.attn_layers[layer].attn_heads]
        weights = torch.cat(weights, dim=0) # dimension of (nhead, 6)
        if self.att_bias:
            biases = torch.cat([head.attn_bias.T for head in self.encoder.attn_layers[layer].attn_heads], dim=0)
            return torch.cat([weights, biases], dim=0)
        return weights

class Encoder(nn.Module):
    ''' Encapsulates multiple layers of MultiHeadAttention and Dense Feed forward networks'''
    def __init__(self, nlayers, nhead, d_model, d_k, d_v, dropout=0.1, dense=False, att_type="scaled_dot", 
                 ablate_learnt_att=False, ablate_att=False, att_weight_reg=False, att_bias=False):
        super().__init__()
        
        # Attention and dense layers
        self.nlayers = nlayers
        attn_layers = [MultiHeadAttention(nhead, d_model, d_k, d_v, dropout=dropout, dense=dense, 
                            att_type=att_type, ablate_learnt_att=ablate_learnt_att, ablate_att=ablate_att, att_weight_reg=att_weight_reg, att_bias=att_bias) 
                       for _ in range(nlayers)]
        self.attn_layers = torch.nn.ModuleList(attn_layers) # properly registers the list of modules, so that .cuda() converts them all to cuda
        self.dense = dense
        if dense:
            self.dense_layers = torch.nn.ModuleList([Dense(d_model, dropout=dropout) for _ in range(nlayers)])
        
        # Final residual connection (across multiple layers)
        self.fc = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, samples, N, K, query=True):
        original = samples.clone()
        attentions = []
        for i in range(self.nlayers):
            samples, attn, pre_residual = self.attn_layers[i](samples, samples, samples, N, K, query=query)
            attentions.append(attn)
            if self.dense:
                samples = self.dense_layers[i](samples)
        
        if self.nlayers > 1:
            samples = self.fc(samples)
            samples = self.layer_norm(samples + original)
        return samples, attentions, pre_residual # return only the last pre_residual, for the moment we only care about visualising a 1 layer model
    
class Dense(nn.Module):
    '''The standard attention is all you need FFN with activation function'''
    
    def __init__(self, d_model, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.relu = torch.nn.ReLU()
        self.fc2 = nn.Linear(d_model, d_model)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X):
        output = self.fc2(self.relu(self.fc1(X)))
        output = self.dropout(output)
        output = self.layer_norm(output + X) # residual connection
        return output

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module with residual sum built in'''
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dense=False, att_type="scaled_dot", 
                 ablate_learnt_att=False, ablate_att=False, att_weight_reg=False, att_bias=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.ablate_att = ablate_att

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        # Multiple attention heads, each with different learnt attention weighs
        # setting temperature = np.sqrt(d_k) enables scaled dot product attention
        self.attn_heads = torch.nn.ModuleList([WeightedAttention(temperature=np.sqrt(d_k), dropout=dropout,
            att_type=att_type, ablate_learnt_att=ablate_learnt_att, att_weight_reg=att_weight_reg, att_bias=att_bias) for _ in range(self.n_head)])
        #self.attn_heads = torch.nn.ModuleList([VectorWeightedAttention(d_model, temperature=np.sqrt(d_k), dropout=dropout, 
        #    att_type=att_type, ablate_learnt_att=ablate_learnt_att) for _ in range(self.n_head)])
        #self.attn_heads = torch.nn.ModuleList([ScaledDotProductRelativeAttention(d_model, temperature=np.sqrt(d_k), dropout=dropout) for _ in range(self.n_head)])
        #self.attn_heads = torch.nn.ModuleList([FixedAttention(temperature=np.sqrt(d_k), dropout=dropout, ablate_learnt_att=ablate_learnt_att) for _ in range(self.n_head)])
        
        # for the residual connection
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # optional feed forward layer
        self.dense = dense
        self.fc = nn.Linear(d_model, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        
    def forward(self, q, k, v, N, K, query=True):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # apply linear transformations
        residual = q.clone()
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # transpose to make the head dimension first, then calculate attention for each head
        if not self.ablate_att:
            q = q.permute(2, 0, 1, 3).contiguous()
            k = k.permute(2, 0, 1, 3).contiguous()
            v = v.permute(2, 0, 1, 3).contiguous()
            head_outputs = []
            for i in range(n_head):
                output, attn = self.attn_heads[i](q[i], k[i], v[i], N, K, query=query)
                head_outputs.append(output)
            output = torch.cat(head_outputs, dim=-1).view(sz_b, len_q, self.d_model) # concat and resize to (batch, sequence_length, dimension)
            attn = attn.mean(dim=0) # take the mean of the attentions over the queries so we can plot them, TODO: update to work with multiple attention heads
        else:
            output = q.view(sz_b, len_q, self.d_model)
            attn = torch.eye(len_q, len_q) # define a dummy matrix to avoid errors
            
        # dropout and add residual
        if not self.dense: # only use this fully connected layer if we're not already using dense layers
            output = self.fc(output)
        pre_residual = self.dropout(output)
        output = self.layer_norm(pre_residual + residual)
        return output, attn, pre_residual

class WeightedAttention(nn.Module):
    ''' Implements various forms of weighted attention'''
    def __init__(self, temperature, dropout=0.1, att_type="scaled_dot", ablate_learnt_att=False, absolute_init=False, att_weight_reg=False, att_bias=False):
        super().__init__()
        self.temperature = temperature
        self.att_type = att_type
        self.ablate_learnt_att = ablate_learnt_att
        self.att_weight_reg = att_weight_reg
        self.absolute_init = absolute_init # initialize to mostly positive values and divide weights by the L1 norm to prevent decay
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        
        # Create the attention weight parameters
        if self.absolute_init:
            self.attn_weight = torch.nn.Parameter((torch.rand(6)*torch.tensor([1, 1, -1, 1, 1, 1])).reshape(6, 1), requires_grad=True)
        elif self.att_weight_reg:
            self.attn_weight = torch.nn.Parameter((torch.tensor([1, 1, 1, 1, 1, 1]).float()).reshape(6, 1), requires_grad=True)
        else:
            self.attn_weight = torch.nn.Parameter(torch.empty(6, 1), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.attn_weight, gain=torch.nn.init.calculate_gain('tanh')) # we can use tanh gain or a lower one with gain=0.1
        
        # Other initialisation methods
        # self.attn_weight = torch.nn.Parameter(torch.abs(self.attn_weight), requires_grad=True) # all parameters can start positive
        # self.attn_weight = torch.nn.Parameter(torch.tensor([1, 0.5, -0.5, 0.5, 0.2, 1]), requires_grad=True) # hypothetical optimal initialisation
        # self.attn_weight = torch.nn.Parameter(torch.tensor([1, 1, 0, 0, 0, 1]).double(), requires_grad=True) # initialise to prototypical networks
        # torch.nn.init.xavier_uniform_(self.attn_bias, gain=torch.nn.init.calculate_gain('tanh'))
        
        # Optionally can use a bias as an addition or the sole weight
        self.att_bias = att_bias
        self.attn_bias = torch.nn.Parameter(torch.zeros(6, 1), requires_grad=True)

    # create the attention weight matrix for an N-way K-shot problem
    def get_attn_weight(self, N, K, query=True, bias=False):
        if not bias: # i.e. this function is called externally
            weight_vector = self.attn_weight
        else:
            weight_vector = self.attn_bias
        if self.absolute_init:
            weight_vector = 6 * weight_vector / torch.linalg.norm(weight_vector, ord=1, dim=0)
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
        
    def forward(self, q, k, v, N, K, query=True): 
        # compute attention, note each of q, k and v has size (batch_size, seq length, hidden_size)
        batch_size, seq_length, hidden_size = q.shape
        
        if self.att_type == "scaled_dot":
            attn = torch.bmm(q, k.transpose(1, 2))   
            attn = attn / self.temperature
        elif self.att_type == "euclidean": #attn[:, i, j] is the euclidean distance between seq item i and seq item j
            expanded_q = q.unsqueeze(2).expand(batch_size, seq_length, seq_length, hidden_size) # expanding along the third dimension duplicates each seq_item seq_length times
            expanded_k = k.unsqueeze(1).expand(batch_size, seq_length, seq_length, hidden_size) # expanding along the second dimension duplicates the entire matrix seq_length times
            if seq_length <= 100:
                distance = euclid_distance(expanded_q, expanded_k)
            else: # for large batch sizes we need to split up the calculation to reduce memory usage
                distances = []
                for i in range(batch_size):
                    distances.append(euclid_distance(expanded_q[i], expanded_k[i]))
                distance = torch.stack(distances, dim=0)
            attn = -1*distance # multiply by -1 so that low distance = high attention
            attn = attn / self.temperature
        elif self.att_type == "cosine":
            q_norm = q/torch.linalg.norm(q, dim=-1).unsqueeze(-1).expand(batch_size, seq_length, hidden_size)
            k_norm = k/torch.linalg.norm(k, dim=-1).unsqueeze(-1).expand(batch_size, seq_length, hidden_size)
            attn = torch.bmm(q_norm, k_norm.transpose(1, 2))
        else:
            raise ValueError("Attention type \"{}\" is invalid".format(self.att_type))
            
        attn = self.softmax(attn) # has size Total_Q, N*K+1, N*K+1
        
        # weight the attn by using a weight matrix constructed from the learnt weights
        if not self.ablate_learnt_att:
            weights = self.get_attn_weight(N, K, query=query, bias=False)
            if self.att_bias:
                bias = self.get_attn_weight(N, K, query=query, bias=True)
                attn = attn + bias
            if self.att_weight_reg:
                attn = weights * attn # this can work with regularization to keep the weights near 1
            else:
                attn = torch.tanh(weights) * attn # best performer
                
            #attn = self.softmax(attn + weights) # bias before the softmax
            #attn = torch.tanh(weights*attn + bias) # second best performer
            #attn = torch.tanh(weights) * attn + torch.tanh(bias) # mostly just worse
            #attn = attn + bias # little change from prototypical networks
        
        # compute output
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn
    
class VectorWeightedAttention(nn.Module):
    ''' Uses a vector to weight each dimension of the samples when calculating attention'''
    def __init__(self, hidden_size, temperature, dropout=0.1, att_type="scaled_dot", ablate_learnt_att=False):
        super().__init__()
        self.temperature = temperature
        self.att_type = att_type
        self.ablate_learnt_att = ablate_learnt_att
        self.hidden_size = hidden_size
        self.mini_batch_size = 16
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        
        # Create the attention weight parameters
        self.attn_weight = torch.nn.Parameter(torch.empty(6, self.hidden_size), requires_grad=True)
        #torch.nn.init.xavier_uniform_(self.attn_weight, gain=torch.nn.init.calculate_gain('tanh')) # we can use tanh gain or a lower one with gain=0.1
        
        # xavier intialization https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf proposed to normalise the initialization by the receptive field size and the output size
        # However self.attn_weight is not a standard layer, only 1 of the 6 vectors is applied to any given item, and the operation is elementwise multiplication not matrix multiplication
        # so self.attn_weight could be considered a grouping of parameters each with a receptive field and output size of 1, hence we use the following initialization following the xavier formula
        torch.nn.init.uniform_(self.attn_weight, a=-np.sqrt(6/2), b=np.sqrt(6/2)) 
        
        
    # create the attention weight matrix for an N-way K-shot problem
    def get_attn_weight(self, N, K):
        w_same_sample, w_same_class, w_diff_class, w_stoquery, w_querytos, w_query = self.attn_weight
        weights = torch.zeros((N*K + 1, N*K + 1, self.hidden_size)).cuda()
        
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
        # compute attention, note each of q, k and v has size (batch_size, seq length, hidden_size)
        batch_size, seq_length, _ = q.shape
        attn = torch.bmm(q, k.transpose(1, 2))   
        attn = attn / self.temperature
        attn = self.softmax(attn).reshape(batch_size, seq_length, seq_length)
        attn = self.dropout(attn)
        
        # compute the weighting
        weights = self.get_attn_weight(N, K) # shape (seq_length, seq_length, hidden_size)
        if not query:
            weights = weights[:-1, :-1] 

        v = v.unsqueeze(1).expand(batch_size, seq_length, seq_length, self.hidden_size) # adds another dimension that repeats v seq_length times
        if seq_length <= 100:
            v = v*weights # multiply each vector by the relative weights (broadcast along dim=0)
            v = attn.unsqueeze(3).expand(batch_size, seq_length, seq_length, self.hidden_size)*v    # multiply by the attention weights (broadcast along dim=-1)
            output = torch.sum(v, dim=2).reshape(batch_size, seq_length, self.hidden_size) # perform the attn weighted sum) 
        else:
            outputs = []
            for i in range(math.ceil(batch_size/self.mini_batch_size)):
                idx_start = i*self.mini_batch_size
                idx_end = min((i+1)*self.mini_batch_size, batch_size)
                minibatch_size = idx_end - idx_start
                v_minibatch = v[idx_start:idx_end]
                
                v_minibatch = v_minibatch * weights
                #print(i, attn.shape)
                attn_minibatch = attn[idx_start:idx_end].unsqueeze(2).reshape(minibatch_size, seq_length, 1 , seq_length)
                output = torch.bmm(attn_minibatch.reshape(-1, 1, seq_length), v_minibatch.reshape(-1, seq_length, self.hidden_size))
                outputs.append(output.reshape(minibatch_size, seq_length, self.hidden_size))
            output = torch.cat(outputs, dim=0)
            
        return output, attn
    
class ScaledDotProductRelativeAttention(nn.Module):
    """An implementation of the relative position representation presented here https://arxiv.org/pdf/1803.02155.pdf """
    def __init__(self, hidden_size, temperature, dropout=0.1, ablate_learnt_att=False):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        self.hidden_size = hidden_size
        
        # Create the relative embeddings for the keys, we want 6 as we have the same 6 types of relations
        self.key_relative_embedding = torch.nn.Parameter(torch.empty(6, self.hidden_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.key_relative_embedding, gain=torch.nn.init.calculate_gain('tanh')) # we can use tanh gain or a lower one with gain=0.1
        self.attn_weight = torch.nn.Parameter(torch.zeros(6, 1), requires_grad=True) # todo: remove this, it is currently printed during training

    # create the relative embedding matrix of the keys for an N-way K-shot problem
    def get_key_matrix(self, N, K):
        key_same_sample, key_same_class, key_diff_class, key_stoquery, key_querytos, key_query = self.key_relative_embedding
        key_matrix = torch.zeros((N*K + 1, N*K + 1, self.hidden_size)).cuda()
        
        # Set the weights for the support set
        for row in range(N*K):
            key_matrix[row, :, :] = key_diff_class
            key_matrix[row, -1, :] = key_stoquery
            key_matrix[row, int(np.floor(row/K)*K):int(np.ceil((row+1)/K)*K), :] = key_same_class
            key_matrix[row, row, :] = key_same_sample
            
        # Set the weights for the query, it must always be the last item in the sequence 
        key_matrix[-1, :-1] = key_querytos
        key_matrix[-1, -1] = key_query
        return key_matrix
        
    def forward(self, q, k, v, N, K, query=True): 
        # get key matrix, truncuate if there is no query
        key_matrix = self.get_key_matrix(N, K)
        if not query:
            key_matrix = key_matrix[:-1, :-1, :]
        
        # compute attention, q has shape (total_Q, N*K+1, hidden_size), key_matrix (N*K+1, N*K+1, hidden_size)
        attn = torch.bmm(q, k.transpose(1, 2)) # shape (total_Q, N*K+1, N*K+1)
        # below has size of (N*K+1, total_Q, N*K+1), where the [0, 1, 2] element represents the attention of the 1th batch's 0th query with the 2th key
        relative_position_term = torch.bmm(q.transpose(0, 1), key_matrix.transpose(1, 2)) 
        attn = attn + relative_position_term.transpose(0, 1) # shift batch size back to the first dimension
        attn = attn / self.temperature
        attn = self.softmax(attn) # has size Total_Q, N*K+1, N*K+1
        
        # compute output
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class FixedAttention(nn.Module):
    ''' 
    Attention except it is fixed, only using the learnt att weights. 
    So attention is calculated purely based on the relation between samples of different classes/queries
    e.g. samples from the other classes all have the same weighted contribution attn_weight[2] to the adapted sample
    This allows a direct learning of prototypical networks, and possibly the ability to learn a better aggregation
    '''
    def __init__(self, temperature, dropout=0.1, ablate_learnt_att=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        
        # Create the attention weight parameters
        self.attn_weight = torch.nn.Parameter(torch.tensor([1, 1, 0, 0, 0, 1]).double(), requires_grad=(not ablate_learnt_att)) # initialise to prototypical networks

    # create the attention weight matrix for an N-way K-shot problem
    def get_attn_weight(self, N, K, weight_vector=None):
        if weight_vector is None: # i.e. this function is called externally
            weight_vector = self.attn_weight
        w_same_sample, w_same_class, w_diff_class, w_stoquery, w_querytos, w_query = weight_vector
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
        attn = self.get_attn_weight(N, K, weight_vector=self.attn_weight)
        total_samples = N*K + 1
        if not query: # during regularisation no query is provided, in which case we simply trim the weight matrix to remove the query related weights
            attn = attn[:-1, :-1]
            total_samples -= 1
            
        # We want to normalize the attention, so that shifting from 10-Way 5-shot to 59-way 5-shot doesn't result in an increase in the activations and potential performance degradation
        # But we don't want to use softmax, so that we can have negative attention
        # So we settle for dividing by the L1 norm
        norm = torch.linalg.norm(attn, ord=1, dim=1) # the norm for each row
        attn = attn/norm
        
        # compute output, duplicating attn along the batch dimension so we can do batch matmul
        attn = self.dropout(attn)
        batch_size = q.shape[0]
        attn = attn.unsqueeze(0).expand(batch_size, total_samples, total_samples)
        output = torch.bmm(attn, v)
        return output, attn
    
    
# Index based implementation: faster but it uses too much memory
class VectorWeightedAttentionIdxVersion(nn.Module):
    ''' Uses a vector to weight each dimension of the samples when calculating attention'''
    def __init__(self, hidden_size, temperature, dropout=0.1, att_type="scaled_dot", ablate_learnt_att=False):
        super().__init__()
        self.temperature = temperature
        self.att_type = att_type
        self.ablate_learnt_att = ablate_learnt_att
        self.hidden_size = hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        
        # Create the attention weight parameters
        self.attn_weight = torch.nn.Parameter(torch.empty(6, self.hidden_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.attn_weight, gain=torch.nn.init.calculate_gain('tanh')) # we can use tanh gain or a lower one with gain=0.1

    # create a matrix to store the indexes of the self.attn_weight vector that we should multiply each value by
    def get_attn_weight_idx(self, N, K, batch_size, query=True):
        # Set the index for the support set
        seq_length = N*K+1 if query else N*K
        index = torch.zeros((seq_length, seq_length)).long().cuda()
        for row in range(N*K):
            index[row, :] = 2 # diff class weight
            index[row, -1] = 3 # query weight
            index[row, int(np.floor(row/K)*K):int(np.ceil((row+1)/K)*K)] = 1 # same class weight
            index[row, row] = 0 # same sample weight
            
        # Set the weights for the query, it must always be the last item in the sequence 
        if query:
            index[-1, :-1] = 4 # w_querytos
            index[-1, -1] = 5 # w_query
        
        # flatten the index
        # first is batch size index, counts from 0 to batch_size-1, repeating each number seq_length*seq_length times
        axis0idx = torch.arange(batch_size).reshape(batch_size, 1).repeat(1, seq_length*seq_length).reshape(-1)
        axis1idx = torch.arange(seq_length).reshape(seq_length, 1).repeat(1, seq_length).reshape(-1).repeat(batch_size) # index for each row
        axis2idx = torch.arange(seq_length).repeat(seq_length).repeat(batch_size) # index for each column
        axis3idx = index.reshape(seq_length*seq_length).repeat(batch_size)
        return axis0idx, axis1idx, axis2idx, axis3idx
        
    def forward(self, q, k, v, N, K, query=True): 
        # compute attention, note each of q, k and v has size (batch_size, seq length, hidden_size)
        batch_size, seq_length, _ = q.shape
        attn = torch.bmm(q, k.transpose(1, 2))   
        attn = attn / self.temperature
        attn = self.softmax(attn).reshape(batch_size, seq_length, seq_length)
        attn = self.dropout(attn)
        
        # compute the weighting
        axis0idx, axis1idx, axis2idx, axis3idx = self.get_attn_weight_idx(N, K, batch_size, query=query)

        # multiply each vector by each of the 6 weights
        v = v.unsqueeze(2).expand(batch_size, seq_length, 6, self.hidden_size)
        v = v*self.attn_weight # multiply each vector by the relative weights (broadcast along dim=0)
        
        # Select the from the 6 versions of each vector to calculate attention
        v = v.unsqueeze(1).expand(batch_size, seq_length, seq_length, 6, self.hidden_size)
        v_weighted = v[axis0idx, axis1idx, axis2idx, axis3idx, :].reshape(batch_size, seq_length, seq_length, self.hidden_size)
        v_weighted = attn.unsqueeze(3).expand(batch_size, seq_length, seq_length, self.hidden_size)*v_weighted    # multiply by the attention weights (broadcast along dim=-1)
        output = torch.sum(v_weighted, dim=2).reshape(batch_size, seq_length, self.hidden_size) # perform the attn weighted sum)   
            
        return output, attn