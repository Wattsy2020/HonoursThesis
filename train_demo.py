from fewshot_re_kit.data_loader import get_loader, get_loader_unsupervised, get_loader_test
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, TinyBERTSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, RobertaPAIRSentenceEncoder
import models
from models.proto import Proto
from models.proto_hatt import ProtoHATT
from models.gnn import GNN
from models.snail import SNAIL
from models.metanet import MetaNet
from models.siamese import Siamese
from models.pair import Pair
from models.d import Discriminator
from models.mtb import Mtb
from models.metaformer_old2 import MetaFormer
from models.metaformer_query import MetaFormerQuery
from models.metaformer_core import MetaFormerCore
from models.metaformer_proto import MetaFormerProto
from models.feat import FEAT
from models.att_proto import att_proto
import sys
import torch
from torch import optim, nn
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import zipfile
import warnings

def main():
    parser = argparse.ArgumentParser()
    
    # Training arguments
    parser.add_argument('--train', default='train_wiki', help='train file')
    parser.add_argument('--val', default='val_wiki', help='val file')
    parser.add_argument('--test', default='test_wiki', help='test file')
    parser.add_argument('--adv', default=None, help='adv file (allows for adversarial training)')
    parser.add_argument('--trainN', default=10, type=int, help='N in train')
    parser.add_argument('--evalN', default=10, type=int, help='N in evaluation')
    parser.add_argument('--testN', default=10, type=int, help='N in testing')
    parser.add_argument('--K', default=5, type=int, help='K shot')
    parser.add_argument('--Q', default=5, type=int, help='Num of query per class')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int, help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int, help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int, help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int, help='val after training how many iters')
    parser.add_argument('--mask_entity', action='store_true', help='mask entity names during training')
    
    # Model arguments
    parser.add_argument('--model', default='proto', help='model name')
    parser.add_argument('--encoder', default='cnn', help='encoder: cnn or bert or roberta or tinybert')
    parser.add_argument('--max_length', default=128, type=int, help='max length')
    parser.add_argument('--load_ckpt', default=None, help='load ckpt')
    parser.add_argument('--save_ckpt', default=None, help='save ckpt')  
    parser.add_argument('--ckpt_name', type=str, default='', help='checkpoint name.')

    # Optimization arguments
    parser.add_argument('--lr', default=-1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int, help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int, help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd', help='sgd / adam / adamw / rmsprop')
    parser.add_argument('--hidden_size', default=230, type=int, help='hidden size')
    parser.add_argument('--n_workers', default=8, type=int, # needed to tackle this bug https://discuss.pytorch.org/t/pytorch-windows-eoferror-ran-out-of-input-when-num-workers-0/25918/3
           help='set the number of workers for the dataloader') 
    parser.add_argument('--fp16', action='store_true', help='use nvidia apex fp16')
    
    # Testing arguments
    parser.add_argument('--only_test', action='store_true', help='only test')
    parser.add_argument('--full_test', action='store_true', help='if enabled test across all settings')
    parser.add_argument('--base_novel', action='store_true', help='if enabled also calculates the separate accuracy on the base and novel classes')
    parser.add_argument('--test_benchmark', action='store_true', help='perform prediction on the unlabelled test set, outputting a zip file containing the results')

    # only for bert / roberta
    parser.add_argument('--pair', action='store_true', help='use pair model')
    parser.add_argument('--pretrain_ckpt', default=None, help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true', help='concatenate entity representation as sentence rep')
    parser.add_argument('--freeze_bert', action='store_true', help='freeze bert when performing ptuning')
    parser.add_argument('--use_sgd_for_bert', action='store_true', help='use SGD instead of AdamW for BERT.')
    
    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', help='use dot instead of L2 distance for proto')
    parser.add_argument('--l2norm', action='store_true', help='apply the l2 normalization to vectors before calculating distance')

    # only for mtb
    parser.add_argument('--no_dropout', action='store_true', help='do not use dropout after BERT (still has dropout in BERT).')
    
    # only for metaformer
    parser.add_argument('--core', action='store_true', help='use the very simplified version of metaformer')
    parser.add_argument('--euclidean', action='store_true', help='use euclidean distance')
    parser.add_argument('--nlayers', default=1, type=int, help='number of transformer layers in the metaformer')
    parser.add_argument('--nheads', default=1, type=int, help='number of heads in the metaformer')
    parser.add_argument('--regularisation', default=0, type=float,  help='the weight of the regularisation term then use the models regularisation objective')
    parser.add_argument('--adapt_query', action='store_true', help='if true adapt the query set alongside the support set')
    parser.add_argument('--dense', action='store_true', help='if true include the dense feedforward network in the metaformer')
    parser.add_argument('--freeze_encoder', action='store_true', # ideally we use a frozen encoder pretrained on this task, 
           help='freeze the sentence encoder while training')    # could act as regularisation and allow the aggregation method to be more robust
    parser.add_argument('--visualisation', action='store_true', help='stores the data before and after metaformer adaptation')
    parser.add_argument('--separate_att_optim', default=None, type=str, help='Uses a second optimizer to optimize the learnt attention weights')
    parser.add_argument('--encoder_lr', default=2e-5, type=float, help='The learning rate of the encoder if it is unfrozen')
    parser.add_argument('--unified_optim', action='store_true', help='use a single rmsprop optimizer for all parameters')
    
    # surrogate novel parameters
    parser.add_argument('--n_surrogate', default=0, type=int, help='number of surrogate novel classes to generate during training')
    parser.add_argument('--n_mtb', default=0, type=int, help='whether to use mtb pretraining to generate surrogate novel classes, and how many classes of it to add')
    
    # metaformer ablation parameters
    parser.add_argument('--ablate_learnt_att', action='store_true', help='remove the learnt attention weights') 
    parser.add_argument('--ablate_att', action='store_true', help='ablate attention in the metaformer entirely')
    parser.add_argument('--att_type', default='scaled_dot', help='attention type') # can also be: cosine, euclidean
    parser.add_argument('--pre_avg', action='store_true', help='average the support set before passing it into the metaformer')
    parser.add_argument('--combined', action='store_true', help='combine both post and pre averaging')
    parser.add_argument('--ablate_mpe_reg', action='store_true', help='remove the mpe regularisation technique')
    parser.add_argument('--ablate_feat_reg', action='store_true', help='ablate the FEAT regularisation')
    parser.add_argument('--att_weight_reg', action='store_true', help='include the attention weight regularisation')
    parser.add_argument('--att_bias', action='store_true', help='include learnable attention bias')
    parser.add_argument('--ablate_query', action='store_true', help='do not perform adaptation on the query')
    
    # for knowledge distillation
    parser.add_argument('--task_distill', action='store_true', help='perform task distillation using a provided teacher and student')
    parser.add_argument('--teacher_ckpt', default=None, help='takes a trained meta-learner')
    parser.add_argument('--encoder_save', default=None, help='the directory to save the student TinyBERT encoder, or any other encoder')
    parser.add_argument('--logit_iter', default=0, type=int, 
        help='the number of iterations to perform knowledge distillation on the logits for (the original paper says it should be much less than intermediate distillation)')
    
    # depreciated hyperparmeters
    parser.add_argument('--kdstrength', default=1, type=float, help='how much of the loss function is made up of kdloss v.s. cross entropy')
    parser.add_argument('--kd_decay', action='store_true', help='if true the weight of the kd loss linearly decrease throughout training')
    parser.add_argument('--nmemory', default=0, type=int, help='number of memory tokens to use')
    parser.add_argument('--attn_style', default='standard', type=str, help='if set to "learnt" uses the learnable attention module')
    
    opt = parser.parse_args()
    trainN = opt.trainN
    evalN = opt.evalN
    testN = opt.testN
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    if opt.n_surrogate > 0 and not opt.freeze_encoder:
        raise ValueError("Generation of surrogate novel classes requires a frozen encoder, so that the distribution of it's embeddings does not change")
    
    print("{}-way-{}-shot Training, {}-way-{}-shot Evaluation, {}-way-{}_shot Testing KD={}".format(trainN, K, evalN, K, testN, K, opt.task_distill))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))
    if opt.task_distill:
        print("Task Specific Knowledge Distillation with: {}".format(opt.teacher_ckpt))
    
    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length)
    elif encoder_name == 'tinybert':
        pretrain_ckpt = opt.pretrain_ckpt or 'pretrain/General_TinyBERT_4L_312D'
        if opt.pair:
            raise NotImplementedError
        else:
            sentence_encoder = TinyBERTSentenceEncoder(
                    pretrain_ckpt,
                    max_length,
                    cat_entity_rep=opt.cat_entity_rep,
                    mask_entity=opt.mask_entity,
                    fit_size=768) # TODO accomodate for fitsize other than 768 if using a model other than BERT
    elif encoder_name == 'prompt':
        pretrain_ckpt = opt.pretrain_ckpt or 'pretrain/General_TinyBERT_4L_312D'
        if opt.pair:
            raise NotImplementedError
        else:
            sentence_encoder = PromptedTinyBERTSentenceEncoder(
                    pretrain_ckpt,
                    max_length,
                    cat_entity_rep=opt.cat_entity_rep,
                    mask_entity=opt.mask_entity,
                    fit_size=768,
                    prompt_length=4, # TODO: add adjustable prompt sizes
                    freeze_bert=opt.freeze_bert) 
    elif encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'pretrain/bert-base-uncased'
        if opt.pair:
            sentence_encoder = BERTPAIRSentenceEncoder(
                    pretrain_ckpt,
                    max_length)
        else:
            sentence_encoder = BERTSentenceEncoder(
                    pretrain_ckpt,
                    max_length,
                    cat_entity_rep=opt.cat_entity_rep,
                    mask_entity=opt.mask_entity)
    elif encoder_name == 'roberta':
        pretrain_ckpt = opt.pretrain_ckpt or 'roberta-base'
        if opt.pair:
            sentence_encoder = RobertaPAIRSentenceEncoder(
                    pretrain_ckpt,
                    max_length)
        else:
            sentence_encoder = RobertaSentenceEncoder(
                    pretrain_ckpt,
                    max_length,
                    cat_entity_rep=opt.cat_entity_rep)
    else:
        raise NotImplementedError
    
    # Create the data loaders, here is where the N and K of the task are decided
    train_data_loader = get_loader(opt.train, sentence_encoder,
            N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, num_workers=opt.n_workers, n_mtb=opt.n_mtb)
    val_data_loader = get_loader(opt.val, sentence_encoder,
            N=evalN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, num_workers=opt.n_workers, incremental=opt.base_novel)
    test_data_loader = get_loader(opt.test, sentence_encoder, N=testN, K=K, Q=Q, na_rate=opt.na_rate, 
            batch_size=batch_size, num_workers=opt.n_workers, incremental=opt.base_novel, visualisation=opt.visualisation)
    if opt.adv:
        adv_data_loader = get_loader_unsupervised(opt.adv, sentence_encoder,
            N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, num_workers=opt.n_workers, incremental=opt.base_novel)
   
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    if opt.adv:
        d = Discriminator(opt.hidden_size)
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, adv_data_loader, adv=opt.adv, d=d)
    else:
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(trainN), str(evalN), str(testN), str(K), "lr={}".format(opt.lr), "dropout={}".format(opt.dropout)])
    if opt.adv is not None:
        prefix += '-adv_' + opt.adv
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.dot:
        prefix += '-dot'
    if opt.l2norm:
        prefix += '-l2norm' # note this wasn't here before the first experiment using it, be careful with checkpoint names
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if opt.freeze_encoder:
        prefix += '-frozen_encoder'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    if 'config' in sentence_encoder.__dict__:
        prefix += '-' + str(sentence_encoder.config['hidden_size'])
    if opt.task_distill:
        prefix += '-KD={}'.format(opt.kdstrength)
        prefix += 'decay={}'.format(opt.kd_decay)
    if model_name == 'metaformer':
        prefix += 'layers={}'.format(opt.nlayers)
        prefix += 'regularisation={}'.format(opt.regularisation)
        prefix += 'adapt_query={}'.format(opt.adapt_query)
        prefix += 'dense={}'.format(opt.dense)
    
    
    if model_name == 'metaformer':
        if opt.adapt_query:
            model = MetaFormerQuery(sentence_encoder, nheads=opt.nheads, nlayers=opt.nlayers, dropout=opt.dropout, euclidean=opt.euclidean, 
                dense=opt.dense, surrogate_novel=opt.n_surrogate > 0, pre_avg=opt.pre_avg, combined=opt.combined, att_type=opt.att_type,
                ablate_learnt_att=opt.ablate_learnt_att, ablate_att=opt.ablate_att, ablate_mpe_reg=opt.ablate_mpe_reg, 
                ablate_feat_reg=opt.ablate_feat_reg, att_weight_reg=opt.att_weight_reg, ablate_query=opt.ablate_query,
                att_bias=opt.att_bias)
        elif opt.core:
            model = MetaFormerCore(sentence_encoder, nheads=opt.nheads, nlayers=opt.nlayers, dropout=opt.dropout,)
        else:
            print("Warning: Using old metaformer")
            model = MetaFormer(sentence_encoder, dropout=opt.dropout)
        # metaformer1 model = MetaFormer(sentence_encoder, nlayers=opt.nlayers, nmemory=opt.nmemory, dropout=opt.dropout, attn_style=opt.attn_style)
    elif model_name == 'metaformer_proto':
        model = MetaFormerProto(sentence_encoder, dropout=opt.dropout)
    elif model_name == 'att_proto':
        model = att_proto(sentence_encoder, dropout=opt.dropout)
    elif model_name == 'FEAT':
        model = FEAT(sentence_encoder)
    elif model_name == 'proto':
        model = Proto(sentence_encoder, dot=opt.dot, l2norm=opt.l2norm)
    elif model_name == 'proto_hatt':
        model = ProtoHATT(sentence_encoder, K)
    elif model_name == 'gnn':
        model = GNN(sentence_encoder, trainN, hidden_size=opt.hidden_size)
    elif model_name == 'snail':
        model = SNAIL(sentence_encoder, trainN, K, hidden_size=opt.hidden_size)
    elif model_name == 'metanet':
        model = MetaNet(trainN, K, sentence_encoder.embedding, max_length)
    elif model_name == 'siamese':
        model = Siamese(sentence_encoder, hidden_size=opt.hidden_size, dropout=opt.dropout)
    elif model_name == 'pair':
        model = Pair(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == 'mtb':
        model = Mtb(sentence_encoder, use_dropout=not opt.no_dropout)
    else:
        raise NotImplementedError
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    prefix = prefix.replace("/", "_") # if we used "incremental/train.json" as a name then the prefix will appear as a folder name when saving, raising an error
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt
        
    model.cuda()
    if opt.freeze_encoder:
        model.freeze_encoder()
    
    # Define optimization parameters
    bert_optim = True if encoder_name in ['albert', 'tinybert', 'bert', 'roberta'] else False
    if opt.lr == -1:
        opt.lr = 2e-5 if bert_optim else 1e-1
            
    # perform knowledge distillation
    if opt.task_distill and not opt.only_test:
        metaformer_kd = model_name == 'metaformer'
        if opt.teacher_ckpt is None:
            raise ValueError('Doing Knowledge distillation but teacher checkpoint not provided')
        if not isinstance(sentence_encoder, TinyBERTSentenceEncoder):
            raise ValueError('Can only use TinyBERT for distillation')
        if opt.student_save is None and not metaformer_kd:
            raise ValueError('The KD student needs a path to be saved in')
        
        # create the teacher model, later pass the checkpoint to framework.task_distill to copy in the parameters
        # note even though we use TinyBERTSentenceEncoder, if we pass it 'bert-base-uncased' it can still run full BERT, while providing the KD functionality necessary
        teacher_encoder = TinyBERTSentenceEncoder(opt.pretrain_ckpt or 'pretrain/bert-base-uncased', max_length, cat_entity_rep=opt.cat_entity_rep, mask_entity=opt.mask_entity)
        teacher_model = Proto(teacher_encoder, dot=opt.dot)
        teacher_model.cuda()
        
        # call framework.task_distill
        framework.task_distill(model, teacher_model, prefix, batch_size, trainN, evalN, K, Q, metaformer_kd=metaformer_kd, kdstrength=opt.kdstrength, kd_decay=opt.kd_decay,
                pytorch_optim=pytorch_optim, load_ckpt=opt.teacher_ckpt, save_ckpt=ckpt, encoder_ckpt=opt.encoder_save,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=opt.pair, 
                train_iter=opt.train_iter, logit_iter=opt.logit_iter, val_iter=opt.val_iter, grad_iter=opt.grad_iter, bert_optim=bert_optim, 
                learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert)
    # perform normal training
    elif not opt.only_test:
        opt.train_iter = opt.train_iter * opt.grad_iter
        #with torch.autograd.profiler.profile(use_cuda=True) as prof: # can perform profiling
        trainN += opt.n_mtb # add the extra mtb classes
        framework.train(model, prefix, batch_size, trainN, evalN, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, encoder_ckpt=opt.encoder_save, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=opt.pair, 
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim, 
                learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert, grad_iter=opt.grad_iter, 
                regularisation = opt.regularisation, n_surrogate=opt.n_surrogate, separate_att_optim=opt.separate_att_optim, 
                encoder_lr=opt.encoder_lr, unified_optim=opt.unified_optim)
        #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'
            
    if opt.visualisation:
        print("Saving the model's hidden states during test data prediction")
        folder = "results/adapted_samples/{}".format(prefix)
        framework.eval(model, batch_size, testN, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair, visualisation=True, result_folder=folder)
        return

    # If we record attentions then create a folder to store them in
    record_att = model_name=="metaformer"
    att_maps_folder = "results/attention_maps/"
    folder = att_maps_folder + prefix
    if record_att:
        if os.path.exists(folder):
            rmtree(folder)
        if not os.path.exists(att_maps_folder):
            os.mkdir(att_maps_folder)
        os.mkdir(folder)
            
    acc = framework.eval(model, batch_size, testN, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair, base_novel=opt.base_novel,
                        record_attention=record_att, result_folder=folder, result_file="default")
    print("TEST SET RESULT: %.2f" % (acc * 100))
        
    if opt.full_test: # test on all 4 settings on the provided test dataset
        for N in [5, 10]:
            for K in [1, 5]:
                if model_name == "proto_hatt" and K != opt.K: continue # proto HATT can only evaluate tasks with the same K as it was trained on

                test_data_loader = get_loader(opt.test, sentence_encoder,
                    N=N, K=K, Q=K, na_rate=opt.na_rate, batch_size=batch_size, num_workers=0)
                framework.test_data_loader = test_data_loader
                acc = framework.eval(model, batch_size, N, K, K, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair,
                                    record_attention=record_att, result_folder=folder, result_file="{}_{}".format(N, K))
                print("Result for {}-{} problem: {:.2f}".format(N, K, acc*100))
    
    #Codalab leaderboard (https://competitions.codalab.org/competitions/27980#participate-submit_results) Submission Requirements: 
    #You should submit 4 files, named as pred-$N-$K.json, for $N in [5, 10] and $K in [1, 5], and directly compress them in one zip (no folder). In each file there is a list:
    #[label0, label1, label2, ...]
    if opt.test_benchmark:
        # Create folder
        folder = "results/{}".format(prefix)
        if not os.path.exists(folder):
            os.mkdir(folder)
        zipf = zipfile.ZipFile(folder +'.zip', 'w', zipfile.ZIP_DEFLATED) # create a zip archive that we will proceed to add files to
        
        print("Making predictions on the test set")
        for N in [5, 10]:
            for K in [1, 5]:
                if model_name == "proto_hatt" and K != opt.K: continue # proto HATT can only evaluate tasks with the same K as it was trained on
                    
                data_loader = get_loader_test("test_wiki_input-{}-{}".format(N, K), sentence_encoder, N, K, 1, batch_size, num_workers=0)
                framework.test_data_loader = data_loader
                preds = framework.test_predict(model, batch_size, N, K, 1, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair)
                preds = [int(x) for x in preds.cpu().numpy()[:, 0]]
                
                # write predictions to file
                path = os.path.join(folder, "pred-{}-{}.json".format(N, K))
                with open(path, "w") as file:
                    json_str = json.dumps(preds)
                    file.write(json_str)
                zipf.write(path, arcname="pred-{}-{}.json".format(N, K))
        zipf.close()
        
    # plot the learnt attention weights embeddings, must be done after evaluation which loads the model if only_test is True
    if opt.model=="metaformer":
        for layer in range(opt.nlayers):
            weights = model.get_attn_weight_matrix(opt.trainN, opt.K, layer)
            plt.matshow(weights.cpu().detach().numpy(), cmap="RdBu")
            plt.colorbar()
            plt.title("Learnt attention weights")
            plt.savefig(os.path.join(folder, "layer{}.png".format(layer)))
        
if __name__ == "__main__":
    main()
