# This file contains custom regularization functions
import torch
import torch.nn.functional as F

# A custom designed regularization loss that maximizes intra-class cosine similarity and minimizes inter-class cosine similarity
def cosine_intra_inter_regularization(center, embed, N, K):
    # Compute distance between the centers and the original embeddings (before passing through set-to-set function)
    # normalize for cosine similarity then calculate similarities between each 2*K examples of N and each of the centers
    center = F.normalize(center, dim=-1)
    embed = F.normalize(embed, dim=-1)
    similarities = torch.matmul(embed, center.T)
    similarities = similarities.view(N, K, N) # i.e. similarities[1, 3, 1] is the distance between class 1 sample 3 and class 1's center

    # Calculate similarity and use it to build the loss
    intra_class_similarity = 0
    inter_class_similarity = 0
    for i_class in range(N):
        intra_similarity = torch.sum(similarities[i_class, :, i_class])
        inter_similarity = torch.sum(similarities[i_class, :, :]) - intra_similarity
        intra_class_similarity += intra_similarity
        inter_class_similarity += inter_similarity

    # Divide by the number of similarities summed to get the mean
    intra_class_similarity /= N*K
    inter_class_similarity /= N*K*(N-1) # we summed the similarities of each N*2*K samples with N-1 other classes

    # Get the loss, it is minimized so we must subtract intra similarity and add inter similarity 
    loss = (1 - intra_class_similarity) + inter_class_similarity
    return loss

# reimplementation of the loss in this paper https://arxiv.org/pdf/2010.16059.pdf
# the main difference from the above is that it uses euclidean distance to calculate the intra_class distance, which could be more useful as cosine similarity tends to always be high within the same class
def mpe_intra_inter_regularization(center, embed, N, K):
    # calculate euclidean distance for each sample with its cluster center
    center_distances = torch.sqrt(torch.sum(torch.pow(embed - center.unsqueeze(1).expand(N, K, -1), 2), dim=-1))
    intra_class_loss = torch.sum(center_distances) / (N*K)

    # calculate cosine similarity between centers
    centers_norm = torch.nn.functional.normalize(center, p=2, dim=-1)
    similarities = torch.matmul(centers_norm, centers_norm.T)
    similarities = similarities[torch.eye(N) == 0] # ignore the distance between a class and itself
    inter_class_loss = torch.sum(similarities)/(N*(N-1)) # there are N*(N-1) similarities calculated
    
    loss = intra_class_loss + 2*inter_class_loss # weight the inter_class_loss a bit higher, as the cosine similarity is restricted to < 1 and so lower than euclidean distance
    return loss

# This encourages the centers post adaptation to be = to the mean of the samples PRE adaptation
# i.e. for the model to be a prototypical network
# It is first outlined here https://arxiv.org/pdf/1812.03664.pdf though their explanation differs from the code here https://github.com/Sha-Lab/FEAT/blob/master/model/models/feat.py
# The regularization outlined in the code is used instead
def mean_loss(center, aux_task, N, K):
    # normalize for cosine similarity
    hidden_size = center.shape[-1]
    center = F.normalize(center, dim=-1)
    aux_task = aux_task.view(N, K, hidden_size)
    center = center.T.unsqueeze(0).expand(N, hidden_size, N).contiguous() # create a batch dimension and duplicate the centers N times, so that torch.bmm works 

    # calculate distances between each K examples of N and each of the centers
    logits_reg = torch.bmm(aux_task, center)
    logits_reg = logits_reg.view(N, K, N) # i.e. logits_reg(1, 3, 1) is the distance between class 1 sample 3 and class 1's center

    # Define labels and calculate loss
    # the support and query set are constructed so that the first K elements belong to class 0, next K elements to class 1 etc... so we can define the label matrix here
    labels = torch.arange(N, dtype=torch.int64).repeat(K).reshape(K, N).T.cuda() # = concatentation of row vectors [i]*(2k) for i in [0, 9] 
    logits_reg = logits_reg.view(N*K, N) # we need to flatten the logits to calculate loss
    labels = labels.contiguous().view(N*K)
    loss = F.cross_entropy(logits_reg, labels)
    return loss
