# Credit: https://github.com/thunlp/HATT-Proto/blob/master/models/proto_hatt.py
import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.nn import functional as F

class ProtoHATT(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, shots):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = sentence_encoder.hidden_size
        if sentence_encoder.cat_entity_rep:
            self.hidden_size *= 2
        self.drop = nn.Dropout()
        
        # for instance-level attention
        self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
        # for feature-level attention
        self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S, Q.unsqueeze(2), 3, score)

    def forward(self, support, query, N, K, total_Q, is_student=False, is_teacher=False):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support_output = self.sentence_encoder(support, is_student=is_student, is_teacher=is_teacher) # (B * N * K, D), where D is the hidden size
        query_output = self.sentence_encoder(query, is_student=is_student, is_teacher=is_teacher) # (B * N * Q, D)
        support = support_output["encoding"].view(-1, N, K, self.hidden_size) # (B, N, K, D)
        query = query_output["encoding"].view(-1, total_Q, self.hidden_size) # (B, N * Q, D)
        #print(support.shape, query.shape)
        
        B = support.size(0) # Batch size
        NQ = query.size(1) # Num of instances for each batch in the query set
  
        # feature-level attention
        fea_att_score = support.view(B * N, 1, K, self.hidden_size) # (B * N, 1, K, D)
        fea_att_score = F.relu(self.conv1(fea_att_score)) # (B * N, 32, K, D) 
        fea_att_score = F.relu(self.conv2(fea_att_score)) # (B * N, 64, K, D)
        fea_att_score = self.drop(fea_att_score)
        fea_att_score = self.conv_final(fea_att_score) # (B * N, 1, 1, D)
        fea_att_score = F.relu(fea_att_score)
        fea_att_score = fea_att_score.view(B, N, self.hidden_size).unsqueeze(1) # (B, 1, N, D)

        # instance-level attention 
        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1) # (B, NQ, N, K, D)
        support_for_att = self.fc(support) 
        query_for_att = self.fc(query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1))
        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1) # (B, NQ, N, K)
        support_proto = (support * ins_att_score.unsqueeze(4).expand(-1, -1, -1, -1, self.hidden_size)).sum(3) # (B, NQ, N, D)

        # Prototypical Networks 
        logits = -self.__batch_dist__(support_proto, query, fea_att_score)
        _, pred = torch.max(logits.view(-1, N), 1)
        
        if is_student or is_teacher:
            return logits, pred, support_output, query_output
        return logits, pred