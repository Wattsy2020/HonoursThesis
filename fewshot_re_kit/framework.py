import os
from shutil import rmtree
import sklearn.metrics
import numpy as np
import sys
import time
import torch
import matplotlib.pyplot as plt
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup

from . import sentence_encoder
from . import data_loader
from . import distribution_calibration

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0
    
# For combining the sequence_outputs and attentions of the support and query set
def concatenate_hidden(output1, output2):
    combined = []
    for layer1, layer2 in zip(output1, output2):
        combined.append(torch.cat([layer1, layer2], dim=0))
    return combined
    
 # A loss that aims to minimise intra class distance and maximise inter class distance
def inter_intra_prototype_loss(support, prototypes):    
      pass
    
# Calculate the distillation_loss between the student and teachers hidden state and attention outputs
# credit: https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT
loss_mse = torch.nn.MSELoss()
layer_weight = [0, 0, 0, 0, 1] # the weight of each layer in the loss function
att_weight = [0, 0, 0, 1]
def hidden_distillation_loss(student_reps, student_atts, teacher_reps, teacher_atts):
    # Define mapping between student and teacher layers
    teacher_layer_num = len(teacher_atts)
    student_layer_num = len(student_atts)
    assert teacher_layer_num % student_layer_num == 0
    layers_per_block = int(teacher_layer_num / student_layer_num)
    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1] for i in range(student_layer_num)]
    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]

    att_loss = 0.
    rep_loss = 0.

    # Calculate loss over attention
    for i, (student_att, teacher_att) in enumerate(zip(student_atts, new_teacher_atts)):
        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).cuda(),
                                  student_att)
        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).cuda(),
                                  teacher_att)

        att_loss += att_weight[i]*loss_mse(student_att, teacher_att)

    # Calculate loss from hidden layer outputs
    new_student_reps = student_reps
    for i, (student_rep, teacher_rep) in enumerate(zip(new_student_reps, new_teacher_reps)):
        rep_loss += layer_weight[i]*loss_mse(student_rep, teacher_rep)

    # Get total loss
    loss = rep_loss + att_loss
    return loss

# Soft cross entropy function used for knowledge distillation on the output logits
# credit: https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT
def soft_cross_entropy(predicts, targets):
    # the logits have shape # (Batch size, number of queries, number of classes)
    student_likelihood = torch.nn.functional.log_softmax(predicts[0, :, :], dim=-1)
    targets_prob = torch.nn.functional.softmax(targets[0, :, :], dim=-1)
    return (- targets_prob * student_likelihood).mean()

# Knowledge distillation for metaformer, with the aim of teaching it how to act as a prototypical network
# This involves aligning the attention weights, the hidden states with the prototypes, and the query with the query calculated by proto net
def metaformer_kd_loss(hidden_states, att_weights, prototypes, query, N, K, nmemory):
    # Generate ideal attention matrix (with 1/N weight for each sample of the same class, 0 weight for everything else)
    seq_length, batch_size, d = hidden_states[0].shape
    ideal_att = torch.zeros(seq_length, seq_length).cuda()
    for row in range(seq_length - 3): # -3 ignores the query and markers
        if row % (K+2) == 0 or (row+1) % (K+2) == 0: # skip the marker rows
            continue 
        start_class = int(np.floor(row/(K+2))*(K+2) + 1) # position of the first example of this class
        end_class = int(np.ceil(row/(K+2))*(K+2) - 2)    # position of last example of this class
        ideal_att[row, start_class:(end_class+1)] = 1/K
    ideal_att = ideal_att.expand(batch_size, seq_length, seq_length) # expand so that direct comparison can be made with the models attention weights
        
    # Create a mask to select the relevant rows of attention weights, so that we don't force the attention for the query or class_markers to 0
    att_row_mask = torch.sum(ideal_att, dim=0) > 0
    att_row_mask = att_row_mask.expand(seq_length, seq_length).T # expand it so that the full attention matrix is masked
    att_row_mask = att_row_mask.expand(batch_size, seq_length, seq_length)
    
    # Calculate MSE between attention of all layers and the ideal matrix
    att_loss = 0.
    for att_matrix in att_weights: # add [-1:] to only do distillation on the last layer
        att_loss += loss_mse(att_matrix[att_row_mask], ideal_att[att_row_mask])
    
    # Calculate MSE between all hidden_states (at the start of the class) and the prototypes
    rep_loss = 0.
    for hidden in hidden_states:
        learner_proto = hidden[torch.arange(N)*(K+2) + 1, :, :] # select prototypes
        learner_proto = learner_proto.transpose(0, 1) # swap to be batch first like the proto net outputs
        assert learner_proto.shape == torch.Size([batch_size, N, d]), "Learner shape: {} Does not match: {}".format(learner_proto.shape, [batch_size, N, d])
        rep_loss += loss_mse(learner_proto, prototypes.expand(batch_size, N, d))
    
    # Calculate MSE between query and query hidden_states
    for hidden in hidden_states:
        learner_query = hidden[-2 - nmemory, :, :]
        rep_loss += loss_mse(learner_query, query[0, :, :]) # note query has shape (1, batch_size, d) so we must reduce it to 2 dimensional
    
    loss = att_loss + rep_loss
    return loss
    
# An abstract class that all meta-learners inherit from
class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(my_sentence_encoder).cuda()
        self.cost = nn.CrossEntropyLoss()
        
    # freeze the encoder's weights so it is not learnt during training
    def freeze_encoder(self):
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError
        
    # performs forward propagation and returns the intermediate outputs (concatenating the query and support set together)
    def forward_distill(self, support, query, N, K, total_Q, is_student=False, is_teacher=False):
        logits, pred, support_output, query_output = self.forward(support, query, N, K, total_Q, is_student=is_student, is_teacher=is_teacher)
        sequence_outputs = concatenate_hidden(support_output["hidden_reps"], query_output["hidden_reps"])
        attentions = concatenate_hidden(support_output["attentions"], query_output["attentions"])
        return logits, pred, sequence_outputs, attentions

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))
    
    # Return: (accuracy, accuracy for queries that belong to base classes, accuracy for queries belonging to novel classes)
    def accuracy_base_novel(self, pred, label, label_names, base_labels, novel_labels):
        pred = pred.view(-1).cpu().numpy()
        label = label.view(-1).cpu().numpy()
        accuracy = np.mean(pred == label)
        
        # separate the base and novel queries
        base_samples_idx = np.array([i for i, class_name in enumerate(label_names) if class_name in base_labels])
        novel_samples_idx = np.array([i for i, class_name in enumerate(label_names) if class_name in novel_labels])
        
        base_accuracy = np.mean(pred[base_samples_idx] == label[base_samples_idx])
        novel_accuracy = np.mean(pred[novel_samples_idx] == label[novel_samples_idx])
        return accuracy, base_accuracy, novel_accuracy

class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv
        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.d = d
            self.d.cuda()
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        elif os.path.isfile("checkpoint/{}".format(ckpt)):
            checkpoint = torch.load("checkpoint/{}".format(ckpt))
            print("Successfully loaded checkpoint '%s'" % "checkpoint/{}".format(ckpt))
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    # Takes a pre-trained teacher meta-learner and a pre-trained TinyBERT (with general distillation)
    # Then performs task specific distillation
    # By copying the teacher meta-learner weights (apart form BERT) into the TinyBERT meta-learner
    # Then freezing the weights of the teacher and TinyBERT meta-learner
    # Then using the knowledge distillation loss to match TinyBERT to BERT
    # Parameters: the same as self.train() except with student and teacher at the start instead of a single model
    def task_distill(self,
            student, teacher,
            model_name,
            B, N_for_train, N_for_eval, K, Q,
            metaformer_kd=False, # if true performs knowledge distillation to train a metaformer to match a prototypical network
            kdstrength = 1, # how strong kd is as a percentage of the loss function (1=100%)
            kd_decay=False, # if true the kdstrength reduces linearly, reaching 0 after half the training iterations
            na_rate=0,
            learning_rate=1e-1,
            lr_step_size=20000,
            weight_decay=1e-5,
            train_iter=30000,
            logit_iter=0, # number of iterations to perform output logit kd for
            val_iter=1000,
            val_step=2000,
            test_iter=3000,
            load_ckpt=None,
            save_ckpt=None,
            encoder_ckpt=None, # where to save the student encoder
            pytorch_optim=optim.SGD,
            bert_optim=False,
            warmup=True,
            warmup_step=300,
            grad_iter=1,
            fp16=False,
            pair=False,
            adv_dis_lr=1e-1,
            adv_enc_lr=1e-1,
            use_sgd_for_bert=False):
        assert load_ckpt is not None, "The teacher requires a checkpoint"
        print("Start training...")
    
        # Init
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(student.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            if use_sgd_for_bert:
                optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
            else:
                optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
            if self.adv:
                optimizer_encoder = AdamW(parameters_to_optimize, lr=1e-5, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        else:
            optimizer = pytorch_optim(student.parameters(),
                    learning_rate, weight_decay=weight_decay)
            if self.adv:
                optimizer_encoder = pytorch_optim(student.parameters(), lr=adv_enc_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if self.adv:
            optimizer_dis = pytorch_optim(self.d.parameters(), lr=adv_dis_lr)

        # Load the checkpoint into both the teacher and the non encoder parameters into the student
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            teacher_state = teacher.state_dict()
            student_state = student.state_dict()
            for name, param in state_dict.items():
                assert name in teacher_state, "Checkpoint doesn't match model, Offending Parameter: {}".format(name)
                #print('load {} from {}'.format(name, load_ckpt))
                teacher_state[name].copy_(param)
                if not name.startswith("sentence_encoder") and not metaformer_kd:
                    student_state[name].copy_(param)

        # Freeze the teacher model
        for param in teacher.parameters():
            param.requires_grad = False
        
        if fp16:
            from apex import amp
            student, optimizer = amp.initialize(student, optimizer, opt_level='O1')

        student.train()
        if self.adv:
            self.d.train()

        # Training Loop
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        init_kdstrength = kdstrength
        
        for it in range(train_iter + logit_iter):
            # Get training data
            support, query, label, _ = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                label = label.cuda()

            # Pass to teacher and student models and calculate knowledge distillation loss
            if metaformer_kd:
                student_logits, pred, hidden_states, att_weights = student.forward(support, query, N_for_train, K, Q * N_for_train + na_rate * Q, is_student=True)
                with torch.no_grad():
                    teacher_logits, _, prototypes, query = teacher.forward(support, query, N_for_train, K, Q * N_for_train + na_rate * Q, is_meta_teacher=True)
                kdloss = metaformer_kd_loss(hidden_states, att_weights, prototypes, query, N_for_train, K, student.nmemory)
            else: # knowledge distillation to train a sentence encoder
                student_logits, pred, student_reps, student_atts = student.forward_distill(support, query, N_for_train, K, Q * N_for_train + na_rate * Q, is_student=True)
                with torch.no_grad():
                    teacher_logits, _, teacher_reps, teacher_atts = teacher.forward_distill(support, query, N_for_train, K, Q * N_for_train + na_rate * Q, is_teacher=True)

                if it < train_iter: # perform intermediate KD
                    kdloss = hidden_distillation_loss(student_reps, student_atts, teacher_reps, teacher_atts) / float(grad_iter)
                else: # output logit KD
                    kdloss = soft_cross_entropy(student_logits, teacher_logits)
            
            if kd_decay and kdstrength > 0:
                kdstrength -= init_kdstrength*(2/train_iter)
            if kdstrength < 1:
                loss = kdstrength*kdloss + (1-kdstrength)*student.loss(student_logits, label) # add in standard cross entropy
            else:
                loss = kdloss
            
            # Optimize
            right = student.accuracy(pred, label)
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 10)
            
            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Adv part
            if self.adv:
                support_adv = next(self.adv_data_loader)
                if torch.cuda.is_available():
                    for k in support_adv:
                        support_adv[k] = support_adv[k].cuda()

                features_ori = student.sentence_encoder(support)
                features_adv = student.sentence_encoder(support_adv)
                features = torch.cat([features_ori, features_adv], 0) 
                total = features.size(0)
                dis_labels = torch.cat([torch.zeros((total // 2)).long().cuda(),
                    torch.ones((total // 2)).long().cuda()], 0)
                dis_logits = self.d(features)
                loss_dis = self.adv_cost(dis_logits, dis_labels)
                _, pred = dis_logits.max(-1)
                right_dis = float((pred == dis_labels).long().sum()) / float(total)
                
                loss_dis.backward(retain_graph=True)
                optimizer_dis.step()
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()

                loss_encoder = self.adv_cost(dis_logits, 1 - dis_labels)
    
                loss_encoder.backward(retain_graph=True)
                optimizer_encoder.step()
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()

                iter_loss_dis += self.item(loss_dis.data)
                iter_right_dis += right_dis

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            if self.adv:
                sys.stdout.write('Step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}'
                    .format(it + 1, iter_loss / iter_sample, 
                        100 * iter_right / iter_sample,
                        iter_loss_dis / iter_sample,
                        100 * iter_right_dis / iter_sample) + '\r')
            else:
                sys.stdout.write('Step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(student, B, N_for_eval, K, Q, val_iter, 
                        na_rate=na_rate, pair=pair)
                student.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': student.state_dict()}, save_ckpt)
                    # note sentence_encoder = nn.DataParallel(sentence_encoder class), so we need to use .module to access the underlying object
                    if not metaformer_kd:
                        student.sentence_encoder.module.save_pretrained(encoder_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)
        
    def train(self,
            model,
            model_name,
            B, N_for_train, N_for_eval, K, Q,
            na_rate=0,
            learning_rate=1e-1,
            lr_step_size=20000,
            weight_decay=1e-5,
            train_iter=30000,
            val_iter=1000,
            val_step=2000,
            test_iter=3000,
            load_ckpt=None,
            save_ckpt=None,
            encoder_ckpt=None,
            pytorch_optim=optim.SGD,
            bert_optim=False,
            warmup=True,
            warmup_step=300,
            grad_iter=1,
            fp16=False,
            pair=False,
            adv_dis_lr=1e-1,
            adv_enc_lr=1e-1,
            regularisation=0, # if > 0 use the model's inbuilt regularisation loss
            use_sgd_for_bert=False,
            n_surrogate=0,
            separate_att_optim=None,
            unified_optim=False, # have a single optimizer for all metaformer parameters
            encoder_lr=2e-5):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        n_surrogate: number of surrogate classes to generate during training
        '''
        print("Start training...")
        if separate_att_optim is not None and unified_optim:
            raise ValueError("Unified optimizer cannot be combined with a separate optimizer")
        
        is_metaformer = "metaformer" in model_name
        encoder_lr = encoder_lr if is_metaformer else learning_rate # metaformer has a separate encoder learning rate parameter
        surrogate = n_surrogate > 0
        if surrogate: 
            if not is_metaformer:
                raise ValueError("Surrogate classes can only be used with metaformer")
            surrogate_generator = distribution_calibration.SurrogateGenerator(n_classes=n_surrogate, k_examples=K+Q, a=0)
    
        # Init
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())
            
            # find parameters for each part of the model
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            learnt_att = ['attn_weight', 'attn_bias'] # add option to train the learnt attention weights with a separate optimizer
            sentence_encoder_params = []
            meta_learner_params = []
            learnt_att_params = []
            for n, p in parameters_to_optimize:
                if "sentence_encoder" in n:
                    sentence_encoder_params.append((n, p))
                elif any(la in n for la in learnt_att):
                    learnt_att_params.append(p)
                else:
                    meta_learner_params.append((n, p))
            
            for name, params in [("sentence encoder", sentence_encoder_params), ("meta learner", meta_learner_params), ("learnt attention", learnt_att_params)]:
                print("\nAll {} parameters:".format(name))
                if name == "learnt attention":
                    print(learnt_att_params)
                else:
                    for n, p in params:
                        print(n)
            
            # Set the parameters specified to have 0 weight decay
            parameters_encoder = [
                {'params': [p for n, p in sentence_encoder_params 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in sentence_encoder_params
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0}
            ]
            parameters_meta_learner = [
                {'params': [p for n, p in meta_learner_params 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in meta_learner_params
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0}
            ]
            parameters_att = [{'params': learnt_att_params, 'weight_decay': 0}]
            
            # Define optimizers, note if the model is not a metaformer, parameters_att will simply be an empty list, so no errors occur
            if separate_att_optim is not None:
                if separate_att_optim == 'sgd':
                    learnt_att_optimizer = torch.optim.SGD(parameters_att, lr=1)
                elif separate_att_optim == 'rmsprop':
                    learnt_att_optimizer = torch.optim.RMSprop(parameters_att, lr=0.001) # this can correct for the low learning rate of each parameter, but it results in weights reducing to 0
                elif separate_att_optim == 'adam':
                    learnt_att_optimizer = AdamW(parameters_encoder, lr=0.1, correct_bias=False) # using adam optimizer doesm't work, the gradients are so small resulting in 0 momentum
                else:
                    raise NotImplementedError
                
                learnt_att_scheduler = get_linear_schedule_with_warmup(learnt_att_optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
                meta_optimizer = torch.optim.SGD(parameters_meta_learner, lr=learning_rate)
            else:
                meta_optimizer = torch.optim.SGD(parameters_meta_learner + parameters_att, lr=learning_rate)
                
            if unified_optim:
                bert_optimizer = torch.optim.RMSprop(parameters_encoder + parameters_att + parameters_meta_learner, lr=learning_rate)
            elif use_sgd_for_bert:
                bert_optimizer = torch.optim.SGD(parameters_encoder, lr=encoder_lr)
            else:
                bert_optimizer = AdamW(parameters_encoder, lr=encoder_lr, correct_bias=False)
            if self.adv:
                optimizer_encoder = AdamW(parameters_encoder, lr=encoder_lr, correct_bias=False)
            
            bert_scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
            meta_scheduler = get_linear_schedule_with_warmup(meta_optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        else:
            optimizer = pytorch_optim(model.parameters(),
                    learning_rate, weight_decay=weight_decay)
            if self.adv:
                optimizer_encoder = pytorch_optim(model.parameters(), lr=adv_enc_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if self.adv:
            optimizer_dis = pytorch_optim(self.d.parameters(), lr=adv_dis_lr)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                #print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            torch.cuda.empty_cache()
            
        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()
        if self.adv:
            self.d.train()
        if "metaformer" in model_name:
            print("Initial Attention Weights:", model.get_attn_weights())
            
        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        start_iter = 0
        for it in range(start_iter, start_iter + train_iter):
            support, query, label, _ = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                label = label.cuda()

            # if using surrogate classes: sample k support examples and Q queries for each of the novel classes
            if surrogate:
                surrogate_classes = next(surrogate_generator)
                surrogate_classes = torch.tensor(surrogate_classes).cuda().float()
                surrogate_support = surrogate_classes[:, :K, :]
                surrogate_query = surrogate_classes[:, K:, :]
                
                # update the labels 
                last_label = label[-1] # labels are a sorted integer tensor, the last class samples are all at the end of the list
                new_label = list(np.arange(n_surrogate).reshape(n_surrogate, 1).repeat(Q, axis=1).reshape(-1)) # e.g. if N=2, and Q = 3 this gives [0, 0, 0, 1, 1, 1]
                new_label = torch.tensor(new_label).long().cuda()
                new_label = new_label + last_label + 1 # e.g. for 10 classes this will transform to [10, 10, 10, 11, 11, 11]
                label = torch.cat([label, new_label]) # finally get the full list of labels [..., 9, 9, 9, 10, 10, 10, 11, 11, 11]
                
                # Predict
                if regularisation > 0:
                    logits, pred, reg_loss = model(support, query, N_for_train, K, Q * N_for_train + na_rate * Q, regularisation=True, surrogate_support=surrogate_support, surrogate_query=surrogate_query)
                else:
                    logits, pred = model(support, query, N_for_train, K, Q * N_for_train + na_rate * Q, surrogate_support=surrogate_support, surrogate_query=surrogate_query)
            else:
                if regularisation > 0:
                    logits, pred, reg_loss = model(support, query, N_for_train, K, Q * N_for_train + na_rate * Q, regularisation=True)
                else:
                    logits, pred = model(support, query, N_for_train, K, Q * N_for_train + na_rate * Q)

            
            loss = model.loss(logits, label)/float(grad_iter)
            if regularisation > 0:
                loss += regularisation*reg_loss/float(grad_iter)
            right = model.accuracy(pred, label)
            
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            
            if it % grad_iter == 0:
                bert_optimizer.step()
                bert_scheduler.step()
                bert_optimizer.zero_grad()
                
                if is_metaformer:
                    meta_optimizer.step()
                    meta_scheduler.step()
                    meta_optimizer.zero_grad()

                    if separate_att_optim is not None:
                        learnt_att_optimizer.step()
                        learnt_att_scheduler.step()
                        learnt_att_optimizer.zero_grad()

            # Adv part
            if self.adv:
                support_adv = next(self.adv_data_loader)
                if torch.cuda.is_available():
                    for k in support_adv:
                        support_adv[k] = support_adv[k].cuda()

                features_ori = model.sentence_encoder(support)
                features_adv = model.sentence_encoder(support_adv)
                features = torch.cat([features_ori, features_adv], 0) 
                total = features.size(0)
                dis_labels = torch.cat([torch.zeros((total // 2)).long().cuda(),
                    torch.ones((total // 2)).long().cuda()], 0)
                dis_logits = self.d(features)
                loss_dis = self.adv_cost(dis_logits, dis_labels)
                _, pred = dis_logits.max(-1)
                right_dis = float((pred == dis_labels).long().sum()) / float(total)
                
                loss_dis.backward(retain_graph=True)
                optimizer_dis.step()
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()

                loss_encoder = self.adv_cost(dis_logits, 1 - dis_labels)
    
                loss_encoder.backward(retain_graph=True)
                optimizer_encoder.step()
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()

                iter_loss_dis += self.item(loss_dis.data)
                iter_right_dis += right_dis

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            if self.adv:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}'
                    .format(it + 1, iter_loss / iter_sample, 
                        100 * iter_right / iter_sample,
                        iter_loss_dis / iter_sample,
                        100 * iter_right_dis / iter_sample) + '\r')
            else:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()
            
            if (it + 1) % val_step == 0:                
                if is_metaformer:
                    sys.stdout.write('\n') # go to a newline so the score isn't overwritten by the next print statements
                    print("Attention Weights", model.get_attn_weights())
                
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter,
                        na_rate=na_rate, pair=pair, record_attention=False) #"metaformer" in save_ckpt, train_iter_prog=it)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                    if encoder_ckpt is not None:
                        model.sentence_encoder.module.save_pretrained(encoder_ckpt)
                if is_metaformer:
                    sys.stdout.write('\n') # go to a newline so the score isn't overwritten by the next print statements
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)

    # Note this uses the testing dataset if calling it with a checkpoint
    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            na_rate=0,
            pair=False,
            ckpt=None,
            base_novel=False,
            record_attention=False,
            result_folder="",
            result_file="",
            train_iter_prog=0,
            visualisation=False): 
        '''
        model: a FewShotREModel instance
        B: Batch size (interestingly enough this is used nowhere)
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        base_novel: If true also print the accuracy for base and novel classes separated
        record_attention: If true record the average attention weights of the metaformer, then create a plot and write it to file
        result_folder: folder to write results to
        result_file: file to write specific result to
        train_iter_prog: if not 0 it records the training iteration at which eval was called
        visualisation: stores data to produce a visualisation of how it is adapted by metaformer
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        raise Exception("Checkpoint not compatible, ckpt parameter {} not found in target model".format(name))
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        # Holds the data for visualisation
        original = []
        pre_residual = []
        adapted = []
        
        # Loop parameters
        iter_right = 0.0
        iter_sample = 0.0
        preds = torch.tensor([]).cuda()
        labels = torch.tensor([]).cuda()
        label_names = []
        attentions = []
        with torch.no_grad():
            for it in range(eval_iter):
                support, query, label, label_name = next(eval_dataset)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()
                if record_attention and len(attentions) < 10:
                    logits, pred, hidden_rep, att_weights = model(support, query, N, K, Q * N + Q * na_rate, is_student=True)
                    attentions.append(att_weights)
                elif visualisation:
                    logits, pred, original_samples, pre_res, samples = model(support, query, N, K, Q * N + Q * na_rate, visualisation=True)
                    
                    # Save memory by shifting the tensors to CPU and only taking 1 example for each batch
                    # This means we only store how the support set is adapted, which is our main goal
                    original.append(original_samples[0, :, :].view(N, K, -1).cpu().detach())
                    pre_residual.append(pre_res[0, :N*K, :].view(N, K, -1).cpu().detach())
                    adapted.append(samples[0, :N*K, :].view(N, K, -1).cpu().detach())
                else:
                    logits, pred = model(support, query, N, K, Q * N + Q * na_rate)
                    
                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_sample += 1
                preds = torch.cat([preds, pred])
                labels = torch.cat([labels, label])
                label_names += label_name
                
                sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                sys.stdout.flush()
            print("")
            
        # write to 3 seperate csv files using pandas
        # stack each list into a single torch tensor, conver to pandas and write
        if visualisation:
            if os.path.exists(result_folder):
                rmtree(result_folder)
            os.mkdir(result_folder)
            
            for sample_array, name in zip([original, pre_residual, adapted], ["original", "pre_residual", "adapted"]):
                sample_tensor = torch.stack(sample_array, dim=0)
                sample_numpy = sample_tensor.numpy()
                
                save_path = os.path.join(result_folder, "{}.npy".format(name))
                np.save(save_path, sample_numpy)
            
        # take the average of attentions over each batch and plot them
        if record_attention:
            # Create/clear a folder to store the results
            if train_iter_prog != 0:
                result_file=str("at train iter: {}".format(train_iter_prog))
            folder = "{}/{}".format(result_folder, result_file)
            if os.path.exists(folder):
                rmtree(folder)
            os.mkdir(folder)
            
            nlayers = len(attentions[0])
            for i in range(nlayers):
                atts = [att[i] for att in attentions] # take the ith layer for each sample
                atts = torch.stack(atts, dim=0)
                seq_length = atts.shape[-1]
                atts = atts.view(-1, seq_length, seq_length)
                avg_attention = torch.mean(atts, dim=0)

                plt.matshow(avg_attention.cpu().detach().numpy(), cmap="Blues")
                plt.colorbar()
                plt.title("Average attention matrix")
                plt.savefig(os.path.join(folder, "layer{}.png".format(i)))
            
        if base_novel:
            # Accuracy across base and novel classes should only be calculated when using the FewRelDataset(..., incremental=True) 
            # so we can retrieve base and novel class lists from the dataset 
            base_labels = set(self.test_data_loader._dataset.base_classes)
            novel_labels = set(self.test_data_loader._dataset.novel_classes)
            
            accuracy, base_acc, novel_acc = model.accuracy_base_novel(preds, labels, label_names, base_labels, novel_labels)
            print("Total Acc: {0:3.2f}% Base Classes: {1:3.2f}% Novel Classes: {2:3.2f}%".format(accuracy*100, base_acc*100, novel_acc*100))

            #print("Attention weights pre tanh: ", model.self_attn.attention.attn_weight)
        
        return iter_right / iter_sample
    
    # Perform prediction on the unlaballed benchmarking test set
    def test_predict(self,
            model,
            B, N, K, Q,
            na_rate=0,
            pair=False,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt != 'none':
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    raise Exception("Checkpoint not compatible")
                own_state[name].copy_(param)
        eval_dataset = self.test_data_loader
        
        preds = []
        with torch.no_grad():
            for it in range(eval_dataset._dataset.length): # we need to iterate through the entire dataset
                if pair:
                    batch = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in batch:
                            batch[k] = batch[k].cuda()
                    logits, pred = model(batch, N, K, 1) # only 1 query per batch
                else:
                    support, query = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                    logits, pred = model(support, query, N, K, 1)
                
                preds.append(pred)
                sys.stdout.write('[Test prediction] step: {0:4}'.format(it + 1) + '\r')
                sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        return torch.stack(preds, dim=0) # concatenate the list of tensors into a single tensor
