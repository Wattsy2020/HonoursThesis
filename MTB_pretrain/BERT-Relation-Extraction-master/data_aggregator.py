# Aggregates the 10 separate datafiles into one JSON dataset
# Additionally it transforms the dataset format to match fewrel
# Where the dataset is made up of a dictionary with keys being the class name and value being a list of dictionaries of token level data, entity locations and entity names
# example: {"P931": 
#    [{
#       "tokens": ["Merpati", "flight", "106", "departed", "Jakarta", "(", "CGK", ")", "on", "a", "domestic", "flight", "to", "Tanjung", "Pandan", "(", "TJQ", ")", "."], 
#       "h": ["tjq", "Q1331049", [[16]]], 
#       "t": ["tanjung pandan", "Q3056359", [[13, 14]]]
#    }, ...
# Finally it masks the entities

# Structure of the pretrain dataset
# A list of tuples: of the form (tuple(list of tokens, location of first entity, location of second entity), first entity name, second entity name)
# Now observe how many entity pairs show up in the dataset (and how often)
import pickle
import json
import random

def write_json_file(json_dict, path):
    with open(path, "w") as file:
        file.write(json.dumps(json_dict))

# REQUIREMENTS: 32GB of memory and SSD/harddrive space for virtual memory depending on these parameters
n_separate_files = 10
min_samples_inner = 10 # the minimum number of samples required to be included as a class 
min_samples_outer = 10
max_samples = 40 # limit the number of samples a single class can have, again to conserve memory
add_masks = False # masking is limited, as we can't mask the normal classes, so perhaps it is not helpful
p_mask = 0.7 # the probability to mask both entities in the relation

json_dataset = {}
class_count = {} # store the size of each class
for i in range(n_separate_files):
    # Load data
    with open("data/D_{}.pkl".format((i+1)*1000), "rb") as file:
        new_data = pickle.load(file)
    print("Converting {} samples from D_{}.pkl".format(len(new_data), (i+1)*1000))

    # add data by entity pair into the json_dataset
    for sample in new_data:
        key = (sample[1], sample[2])
        if key in json_dataset:
            if class_count[key] < max_samples:
                json_dataset[key].append(sample[0])
        else:
            json_dataset[key] = [sample[0]]
        class_count[key] = class_count.get(key, 0) + 1


    # then remove every class from the json_dataset that has less than a small number of samples to save memory
    sorted_classes = sorted(list(class_count.items()), key = lambda s: s[1])
    for class_key, occurences in sorted_classes:
        if occurences >= min_samples_inner: break
        class_count[class_key] = 0
        json_dataset.pop(class_key, None)
    print("Total classes: {}".format(len(json_dataset)))
    del new_data

print("\nCreating Final Dataset\n")
# Remove everything that doesn't appear often enough
# we only need 10 examples but something that appears more often is more likely
# to be a well defined relationship
sorted_classes = sorted(list(class_count.items()), key = lambda s: s[1])
for class_key, occurences in sorted_classes:
    if occurences >= min_samples_outer: break
    json_dataset.pop(class_key, None)
print("Total classes: {}".format(len(json_dataset)))
    
# Finally change the format to match fewrel
keys = list(json_dataset.keys())
for i, key in enumerate(keys):
    old_items = json_dataset[key]
    new_items = []
    
    for item_tuple in old_items:
        tokens = item_tuple[0]
        # Mask the entities
        if add_masks and random.random() < p_mask:
            for j in range(len(tokens)):
                if tokens[j] in key[0] or tokens[j] in key[1]:
                    tokens[j] = "[MASK]"

        # Convert to fewrel
        item_dict = {"tokens": tokens,
            "h": [key[0], "BLANK", [list(item_tuple[1])]], # the middle value is meant to be a Wikidata ID that's not important
            "t": [key[1], "BLANK", [list(item_tuple[2])]]}
        new_items.append(item_dict)

    # create a new key as JSON doesn't support tuple keys
    json_dataset["M{}".format(i)] = new_items
    json_dataset.pop(key, None)

write_json_file(json_dataset, "data/mtb_pretrain.json")
