# Implements an end-to-end meta-learner, instead of predefined aggregation and comparison functions the transformer learns them
import sys
sys.path.append('..')
import fewshot_re_kit
import numpy as np

import torch
from torch import autograd, optim, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.transformer import Transformer

class MetaFormer(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, nlayers=1, nmemory=0, dropout=0.5, act="gelu", attn_style="standard"):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        hidden_size = sentence_encoder.config["hidden_size"]
        self.hidden_size = 2*hidden_size if sentence_encoder.cat_entity_rep else hidden_size # if the entity rep is concatenated the size will be doubled
        self.nmemory = nmemory
        
        # Modules, note dropout is present in self.transformer so no need for another dropout module
        self.sentence_encoder = sentence_encoder
        self.transformer = Transformer(d_model=self.hidden_size, nhead=8, num_encoder_layers=nlayers, 
            dim_feedforward=self.hidden_size, dropout=dropout, activation=act, attn_style=attn_style).cuda()
        self.linear_head = torch.nn.Linear(self.hidden_size, 1).cuda()
        
        # Define the special tokens we use to separate each class' support set in the metaformer
        self.class_start   = torch.nn.Parameter(torch.empty(1, self.hidden_size), requires_grad=True).cuda()
        self.class_end     = torch.nn.Parameter(torch.empty(1, self.hidden_size), requires_grad=True).cuda()
        self.query_marker  = torch.nn.Parameter(torch.empty(1, self.hidden_size), requires_grad=True).cuda()
        self.memory_tokens = torch.nn.Parameter(torch.empty(nmemory, self.hidden_size), requires_grad=True).cuda()
        
        # Initialise them
        torch.nn.init.kaiming_uniform_(self.class_start, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.class_end, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.query_marker, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.memory_tokens, nonlinearity='relu')
        
        # use the below code to check if the special tokens change
        # self.orig = self.class_start.cpu().detach().numpy()
        # print(np.sum(self.orig - self.class_start.cpu().detach().numpy()))
        
    def forward(self, support, query, N, K, total_Q, is_student=False):
        # get encoding vectors
        support = self.sentence_encoder(support)["encoding"]
        query = self.sentence_encoder(query)["encoding"]
        support = support.view(N, K, self.hidden_size) # only a batch size of 1 is supported
        query = query.view(total_Q, 1, self.hidden_size) # we need new dimensions so that we can add the query markers

        # add encoding vectors for the support set to create the input, with the samples for each class being separated by tokens
        # resize the tokens so they can be added to the input
        class_start = torch.clone(self.class_start.view(1, 1, self.hidden_size).expand(N, 1, self.hidden_size)) # expand just copies the same tensor multiple times
        class_end = torch.clone(self.class_end.view(1, 1, self.hidden_size).expand(N, 1, self.hidden_size))
        query_marker = torch.clone(self.query_marker.view(1, 1, self.hidden_size).expand(total_Q, 1, self.hidden_size))

        # use broadcasting to add the tokens in before and after a classes example
        support = torch.cat([class_start, support, class_end], dim=1)
        query = torch.cat([query_marker, query, query_marker], dim=1)
        
        # finally expand and concatenate the support set, query set and memory cell to get the transformer input
        support = torch.clone(support.view(1, N*(K+2), self.hidden_size).expand(total_Q, N*(K+2), self.hidden_size)) # we use the same support set for all queries, so copy it total_Q times
        query = query.view(total_Q, 3, self.hidden_size)
        memory = torch.clone(self.memory_tokens.expand(total_Q, self.nmemory, self.hidden_size))
        input_embeds = torch.cat([support, query, memory], dim=1)

        # pass through the transformer to get representations (which we want to represent a comparison between each class and the query)
        input_embeds = torch.transpose(input_embeds, 0, 1) # swap the axis as the sequence dimension goes first in pytorch 1.7.0
        output = self.transformer(input_embeds, need_hidden=is_student, need_weights=is_student)
        if is_student:
            output, hidden_states, att_weights = output
        output = torch.transpose(output, 0, 1).view(total_Q, N*(K+2) + 3 + self.nmemory, self.hidden_size)
        
        # take representations at the samples and take euclidean distance to get logits for each class
        class_start_index = torch.arange(1, N*(K+2), K+2).cuda() # the indexes of the sample after class_start token in the input
        class_reps = output[:, class_start_index, :].view(total_Q, N, self.hidden_size)
        query_reps = output[:, -2 - self.nmemory, :].view(total_Q, 1, self.hidden_size)
        logits = -torch.sum(torch.pow(class_reps - query_reps, 2), 2) # euclidean distance, note sqrt doesn't change order so not computed, and negative means closer vectors are higher rated
        _, pred = torch.max(logits, dim=1)
        
        if is_student:
            return logits, pred, hidden_states, att_weights
        return logits, pred
