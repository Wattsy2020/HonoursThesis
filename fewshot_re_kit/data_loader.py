import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, incremental=False, nbase_classes=54, visualisation=False, n_mtb=0, p_mask=0.2):
        """
        Initialises the dataset by tokenizing all sentences using encoder.tokenize
        
        Parameters:
            incremental: an option to, when validating/testing, select all base classes + X novel ones to form an episode (matching the methodology of https://aclanthology.org/2020.coling-main.142/)
            nbase_classes: the number of base classes in the incremental setting
            visualisation: only provide examples from a set selection of 10 classes, so we can see how they are adapted by the model
            n_mtb: if more than 0 it adds `n_mtb` classes of pretraining examples generated by matching the blanks, with the goal of adding in surrogate novel classes to each episode
        """
        
        if incremental and visualisation:
            raise ValueError("Cannot visualise with the incremental setting as they both modify class selection logic")
        
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            raise Exception("[ERROR] Data file {}/{} does not exist!".format(root, name))
        with open(path) as file:
            self.json_data = json.load(file) 
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.n_mtb = n_mtb
        self.add_mtb = n_mtb > 0
        
        # tokenize all the sentences in the data, to prevent having to repeat this 1000s of times, removing a bottleneck
        self.items = {} # a dictionary mapping "class" to "list of tuples: (word, pos1, pos2, mask)" (similar to self.json_data, but encoded)
        for class_key in self.json_data:
            for sample in self.json_data[class_key]:
                if class_key not in self.items:
                    self.items[class_key] = []
                
                # randomly mask the tokens if using mtb
                if self.add_mtb and np.random.rand(1) < p_mask: 
                    head = sample['h'][0]
                    tail = sample['t'][0]
                    tokens = sample["tokens"]
                    for i in range(len(tokens)):
                        token = tokens[i].lower()
                        if token in head or token in tail:
                            tokens[i] = "[MASK]"
                    sample["tokens"] = tokens
                    #print(tokens)
                
                self.items[class_key].append(self.__getraw__(sample))
                
        # store the sentences from the mtb training data, we need to tokenize it in the get item function to save memory
        mtb_pretrain_path = "./MTB_pretrain/BERT-Relation-Extraction-master/data/mtb_pretrain.json"
        if self.add_mtb:
            with open(mtb_pretrain_path) as file:
                self.mtb_raw_data = json.load(file) 
            self.mtb_class_iter = 0 # an iterator, so that we sequentially progress through each of the ~30000 classes
            self.total_mtb_classes = len(self.mtb_raw_data)
                
        # The incremental setting is different, as only 5 novel classes are sampled for each episode
        self.nbase = nbase_classes
        self.incremental = incremental
        if incremental: # Load the base and novel classes so we can sample them separately
            with open(os.path.join(root, "incremental/Original/baserel2index.json")) as file:
                self.base_classes = list(json.load(file).keys()) # A list of the base class names
            self.novel_classes = list(set(self.classes) - set(self.base_classes))          
        
        # In the visualisation setting the classes are fixed i.e. the same classes are present in each episode in the same order
        # This is so we can sample data for classes across multiple episodes
        self.visualisation = visualisation
        if visualisation:
            self.target_classes = ['P86', 'P57', 'P750', 'P39', 'P551', 'P84', 'P400', 'P58', 'P463', 'P59'] # 8 base and 2 novel classes
    
    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        if self.incremental:
            assert self.N > self.nbase, "the incremental setting requires classification across all base classes and at least 1 novel class"
            target_classes = self.base_classes + random.sample(self.novel_classes, self.N - self.nbase)
        elif self.visualisation:
            target_classes = self.target_classes
        else:
            target_classes = random.sample(self.classes, self.N)
            
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_label = []
        query_label_name = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        # Interestingly this function doesn't appear to store the class information anywhere, how are the support set classes represented? 
        # Purely through their order as the query label is i, hence K must be passed to all meta learners so they know each block of K samples represents the first class
        # e.g. see this line in proto.py support = support.view(-1, N, K, hidden_size)
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))), 
                self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.items[class_name][j]
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            query_label += [i] * self.Q
            query_label_name += [class_name] * self.Q

        # Optionally add in matching the blanks pretraining examples
        if self.add_mtb:
            for i in range(self.n_mtb):
                class_name = "M{}".format(self.mtb_class_iter)
                self.mtb_class_iter = (self.mtb_class_iter + 1) % self.total_mtb_classes # wrap around when we reach the end of the classes using modulus

                # similar code to above to add in samples to the support and query set
                indices = np.random.choice(
                    list(range(len(self.mtb_raw_data[class_name]))), 
                    self.K + self.Q, False)
                count = 0
                for j in indices:
                    word, pos1, pos2, mask = self.__getraw__(self.mtb_raw_data[class_name][j])
                    word = torch.tensor(word).long()
                    pos1 = torch.tensor(pos1).long()
                    pos2 = torch.tensor(pos2).long()
                    mask = torch.tensor(mask).long()
                    if count < self.K:
                        self.__additem__(support_set, word, pos1, pos2, mask)
                    else:
                        self.__additem__(query_set, word, pos1, pos2, mask)
                    count += 1

                query_label += [self.N + i] * self.Q
                query_label_name += [class_name] * self.Q
            
        return support_set, query_set, query_label, query_label_name
    
    def __len__(self):
        return 1000000000

# This takes a list (batch) of tupples and collates them into one big batch of a dictionary format (batch_support)
def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    batch_label_names = []
    
    support_sets, query_sets, query_labels, query_label_names = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
        batch_label_names += query_label_names[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label, batch_label_names

def get_loader(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data', 
        incremental=False, visualisation=False, n_mtb=0):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root, incremental=incremental, visualisation=visualisation, n_mtb=n_mtb)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

class FewRelUnsupervisedDataset(data.Dataset):
    """
    FewRel Unsupervised Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        total = self.N * self.K
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }

        indices = np.random.choice(list(range(len(self.json_data))), total, False)
        for j in indices:
            word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[j])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(support_set, word, pos1, pos2, mask)

        return support_set
    
    def __len__(self):
        return 1000000000

def collate_fn_unsupervised(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support

def get_loader_unsupervised(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_unsupervised, na_rate=0, root='./data'):
    dataset = FewRelUnsupervisedDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

def flatten(nested_list):
    unnested = []
    for list_ in nested_list:
        unnested += list_
    return unnested


class FewRelTestDataset(data.Dataset):
    """
    A dataset for performing prediction on the test set data which is stored in a different format
    see here: https://competitions.codalab.org/competitions/27980#participate-get_data 
    """
    def __init__(self, name, encoder, N, K, Q, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            raise Exception("[ERROR] Data file {}/{} does not exist!".format(root, name))
        self.json_data = json.load(open(path))
        self.length = len(self.json_data)
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        
        episode = self.json_data[index]
        n_samples = self.N*self.K + 1 # each example has one query
        
        # note that episode["meta_train"] is a two-d list of dimension N, K and needs to be flattened for this to work
        for i, sample in enumerate(flatten(episode["meta_train"]) + [episode["meta_test"]]):
            word, pos1, pos2, mask = self.__getraw__(sample)
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            if i < self.N*self.K:
                self.__additem__(support_set, word, pos1, pos2, mask)
            else:
                self.__additem__(query_set, word, pos1, pos2, mask)

        return support_set, query_set
    
    def __len__(self):
        return self.length

# This takes a list (batch) of tuples and collates them into one big batch of a dictionary format (batch_support)
def collate_fn_test(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets, query_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0) # concatenates all items into a single multi dimensional tensor https://pytorch.org/docs/stable/generated/torch.stack.html 
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    return batch_support, batch_query # the query is already batched

def get_loader_test(name, encoder, N, K, Q, batch_size, num_workers=8, root='./data'):
    dataset = FewRelTestDataset(name, encoder, N, K, Q, root)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True, 
                                  num_workers=num_workers, collate_fn=collate_fn_test)
    return iter(data_loader)


class FewRelSeqDataset(data.Dataset):
    """
    A dataset that loops through the underlying data sequentially, returning examples of the classes in the same order across episodes
    This is useful to store the embeddings of each relation for generation of surrogate novel classes
    """
    def __init__(self, name, encoder, K, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            raise Exception("[ERROR] Data file {}/{} does not exist!".format(root, name))
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.length = int(len(self.json_data[self.classes[0]])/K) # in each iteration we encode K examples for each class 
        self.N = len(self.classes)
        self.K = K
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        
        # for each class, append the next k samples to the support set
        for rclass in self.classes:
            for i, sample in enumerate(self.json_data[rclass][index*self.K:(index+1)*self.K]):
                word, pos1, pos2, mask = self.__getraw__(sample)
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                self.__additem__(support_set, word, pos1, pos2, mask)
    
        return support_set
    
    def __len__(self):
        return self.length

# This takes a list (batch) of the support_sets and collates them into one big batch of a dictionary format (batch_support)
def collate_fn_seq(data):
    support_set = data[0]
    assert type(support_set) == dict 
    
    # concatenate the list of torch.tensors into a single tensor
    for k in support_set:
        support_set[k] = torch.stack(support_set[k], 0)
    return support_set
    
def get_loader_seq(name, encoder, K, batch_size=1, num_workers=8, root='./data'):
    dataset = FewRelSeqDataset(name, encoder, K, root)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True, 
                                  num_workers=num_workers, collate_fn=collate_fn_seq)
    return iter(data_loader)
