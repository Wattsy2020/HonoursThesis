# A batch script to calculate and store vector embeddings for all the relations in the training set using a previously trained sentence encoder
# The aim is to generate surrogate novel classes
from fewshot_re_kit.data_loader import get_loader_seq
from fewshot_re_kit.sentence_encoder import TinyBERTSentenceEncoder
import sys
import torch
import numpy as np
import argparse
import os
import zipfile

def main():
    parser = argparse.ArgumentParser()
    
    # Data settings
    parser.add_argument('--train', default='train_wiki',
            help='train file')
    parser.add_argument('--K', default=1, type=int,
            help='K shot')
    parser.add_argument('--n_workers', default=8, type=int, # needed to tackle this bug https://discuss.pytorch.org/t/pytorch-windows-eoferror-ran-out-of-input-when-num-workers-0/25918/3
           help='set the number of workers for the dataloader') 
    parser.add_argument('--save_name', type=str, default='encodings')
    
    # Encoder settings, note that the encoder should always be tinybert
    parser.add_argument('--encoder', default='tinybert',
            help='encoder: cnn or bert or roberta or tinybert')
    parser.add_argument('--max_length', default=64, type=int,
           help='max length')
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
           help='concatenate entity representation as sentence rep')
    
    opt = parser.parse_args()
    K = opt.K
    batch_size = 1
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    # Create model
    if encoder_name == 'tinybert':
        pretrain_ckpt = opt.pretrain_ckpt or 'pretrain/General_TinyBERT_4L_312D'
        sentence_encoder = TinyBERTSentenceEncoder(
                pretrain_ckpt,
                max_length,
                cat_entity_rep=opt.cat_entity_rep,
                mask_entity=False,
                fit_size=768)
        sentence_encoder.cuda()
    else:
        raise NotImplementedError
    sentence_encoder.eval()
    
    # Create the data loader
    train_data_loader = get_loader_seq(opt.train, sentence_encoder, K=K, batch_size=batch_size, num_workers=opt.n_workers)
   
    # Loop through the data laoder and encode each sample
    encodings = []
    with torch.no_grad():
        for batch in train_data_loader:
            for k in batch:
                batch[k] = batch[k].cuda()
            result = sentence_encoder.forward(batch)["encoding"]
            encodings.append(result.cpu().detach().numpy())

    # Reshape the encodings 
    N = train_data_loader._dataset.N
    batch_size = train_data_loader._dataset.length
    hidden_size = sentence_encoder.config["hidden_size"]
    if opt.cat_entity_rep:
        hidden_size *= 2
    encodings = np.stack(encodings).reshape(batch_size, N, K, hidden_size)
    
    # Write to file
    save_path = "results/{}".format(opt.save_name)
    np.save(save_path, encodings)
        
if __name__ == "__main__":
    main()
