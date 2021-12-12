import math
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from . import network
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, AlbertModel, AlbertTokenizer
from .TinyBERT.transformer.modeling import TinyBert # our addition, import TinyBert for KD

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50, 
            pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, 
                word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, 
                pos_embedding_dim, hidden_size)
        self.word2id = word2id

    def forward(self, inputs, is_student=False, is_teacher=False):
        x = self.embedding(inputs)
        x = self.encoder(x)
        return {"encoding": x}

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1

        return indexed_tokens, pos1, pos2, mask
    
class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity

    def forward(self, inputs, is_student=False, is_teacher=False):
        output = {}
        if not self.cat_entity_rep:
            _, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
            output["encoding"] = x
        else:
            outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            output["encoding"] = state
        return output
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask
    
class BERTPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = BertForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs, is_student=False, is_teacher=False):
        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0]
        return {"encoding": x}
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return indexed_tokens

class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False): 
        nn.Module.__init__(self)
        self.roberta = RobertaModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.cat_entity_rep = cat_entity_rep

    def forward(self, inputs, is_student=False, is_teacher=False):
        output = {}
        if not self.cat_entity_rep:
            _, x = self.roberta(inputs['word'], attention_mask=inputs['mask'])
            output["encoding"] = x
        else:
            outputs = self.roberta(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            output["encoding"] = state
        return output

    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        pE1 = 0
        pE2 = 0
        pE1_ = 0
        pE2_ = 0
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
            if ins[i][1] == E1b:
                pE1 = ins[i][0] + i
            elif ins[i][1] == E2b:
                pE2 = ins[i][0] + i
            elif ins[i][1] == E1e:
                pE1_ = ins[i][0] + i
            else:
                pE2_ = ins[i][0] + i
        pos1_in_index = pE1 + 1
        pos2_in_index = pE2 + 1
        sst = ['<s>'] + sst
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(1)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(sst)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index, pos2_in_index, mask


class RobertaPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.roberta = RobertaForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def forward(self, inputs, is_student=False, is_teacher=False):
        x = self.roberta(inputs['word'], attention_mask=inputs['mask'])[0]
        return {"encoding": x}
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)
        return indexed_tokens 

class TinyBERTSentenceEncoder(nn.Module):
    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False, fit_size=768):
        super().__init__()
        # Store hyperparameters of the model
        with open(os.path.join(os.path.abspath(pretrain_path), 'config.json'), 'r') as file:
            self.config = json.loads(file.read())
        self.hidden_size = self.config['hidden_size']
        
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        
        # if this is BERT (has hidden size 768) and not TinyBERT (hidden size 312) then it won't be able to process a large batch, so we set a limit
        self.max_batch_size = 100 if self.hidden_size > 312 else 500
        print("Max batch size of the sentence encoder: ", self.max_batch_size)
        
        # create a linear layer for scaling outputs up to the teacher models hidden layer size
        self.fit_dense = torch.nn.Linear(self.hidden_size, fit_size) 

    def forward(self, inputs, is_student=False, is_teacher=False):
        batch_size = inputs['word'].shape[0]
        if batch_size > self.max_batch_size and not self.training: # if we're evaluating on a large task we can use BERT by separating the problem into smaller ones to avoid OOM errors
            last_layers = []
            cls_outs = []
            for i in range(math.ceil(batch_size/self.max_batch_size)):
                start_index = i*self.max_batch_size
                stop_index = min((i+1)*self.max_batch_size, batch_size) # either the next max_batch_size elements or all the elements until the end if this is the last batch
                last_layer, cls_out = self.bert(inputs['word'][start_index:stop_index], attention_mask=inputs['mask'][start_index:stop_index])
                last_layers.append(last_layer)
                cls_outs.append(cls_out)
            last_layer = torch.cat(last_layers, dim=0)
            cls_out = torch.cat(cls_outs, dim=0)
        else:
            last_layer, cls_out, all_layers, attentions = self.bert(inputs['word'], attention_mask=inputs['mask'], output_hidden_states=True, output_attentions=True)
        output = {} # store the various outputs in a dictionary
        
        if not self.cat_entity_rep:
            output["encoding"] = cls_out
        else:
            tensor_range = torch.arange(batch_size)
            #print(last_layer.shape, cls_out.shape)
            #print(tensor_range, inputs['pos1'], inputs['pos2'])
            h_state = last_layer[tensor_range, inputs['pos1']] # e.g. last_layer[[1, 2, 3], [10, 3, 7]] will select items [last_layer[1, 10], last_layer[2, 3], last_layer[3, 7]]
            t_state = last_layer[tensor_range, inputs['pos2']]
            output["encoding"] = torch.cat((h_state, t_state), -1)
        if is_student or is_teacher: # return the sequence outputs and attention hidden states to perform knowledge distillation
            # student needs to transform the sequence outputs to have the same dimension as the teacher model
            scaled = [self.fit_dense(layer_out) for layer_out in all_layers] if is_student else all_layers
            output["hidden_reps"] = scaled
            output["attentions"] = attentions
        return output
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask
    
    def save_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.bert.save_pretrained(path)
