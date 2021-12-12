import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn

def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dot=False, l2norm=False):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot
        self.l2norm = l2norm

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    # If this meta learner is a distillation student then also return the outputs of the sentence encoder
    # If the prototypical network is being used to teach the metaformer then it outputs the prototypes and queries
    def forward(self, support, query, N, K, total_Q, is_student=False, is_teacher=False, is_meta_teacher=False):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        if (is_student or is_teacher) and is_meta_teacher:
            raise ValueError("The prototypical network can only perform one type of knowledge distillation at a time")
        
        # Take the output from the TinyBERTSentenceEncoder, which gives a dictionary that can contain hidden states and attentions for KD
        support_output = self.sentence_encoder(support, is_student=is_student, is_teacher=is_teacher)
        query_output = self.sentence_encoder(query, is_student=is_student, is_teacher=is_teacher)
        
        support_emb = support_output["encoding"]
        query_emb = query_output["encoding"] 
        
        hidden_size = support_emb.size(-1)
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        support = support.view(-1, N, K, hidden_size) # (B, N, K, D)
        query = query.view(-1, total_Q, hidden_size) # (B, total_Q, D)

        # Prototypical Networks 
        # Ignore NA policy
        support = torch.mean(support, 2) # Calculate prototype for each class
        if self.l2norm: # if l2norm we have to normalize after the prototypes are calculated
            support_norm = l2norm(support)
            query_norm = l2norm(query)
            logits = self.__batch_dist__(support_norm, query_norm) # (B, total_Q, N)
        else:
            logits = self.__batch_dist__(support, query)
        minn, _ = logits.min(-1) # for some reason takes the class with the lowest probability and then concatenates it to the logits
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        
        if is_student or is_teacher:
            return logits, pred, support_output, query_output
        if is_meta_teacher:
            return logits, pred, support, query
        return logits, pred

    
    
    
