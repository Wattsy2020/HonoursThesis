# combines the train and val sets into one data set that can be used for
# full classification across base + novel classes
import json
import numpy as np
from sklearn.model_selection import train_test_split

def read_json_file(path):
    with open(path, "r") as file:
        json_dict = json.loads(file.read())
    return json_dict

def write_json_file(json_dict, path):
    with open(path, "w") as file:
        file.write(json.dumps(json_dict))

# Add all the data for classes in json2 to json1 and return it
def append_json_data(json1, json2):
    result = {}
    for key, value in json1.items():
        result[key] = value
    for key, value in json2.items():
        result[key] = value
    return result

# Return a train val test split for the required number of classes
def sample_data(source, n_train, n_valid, n_test):
    # Form the sampling indices
    all_idx = np.random.permutation(len(source))
    train_idx = all_idx[:n_train]
    valid_idx = all_idx[n_train:n_train+n_valid]
    test_idx = all_idx[n_train+n_valid:]

    # Create dictionaries using those indices
    train = {}
    valid = {}
    test = {}
    for i, (key, value) in enumerate(source.items()):
        if i in train_idx: train[key] = value
        if i in valid_idx: valid[key] = value
        if i in test_idx: test[key] = value
    return train, valid, test


# Read data
train = read_json_file("train_wiki.json")
valid = read_json_file("val_wiki.json")

# Form a train, validation and test set through combining all data and splitting it
# To do: consider splitting the train set by samples as well (so the test set
# ... has unknown samples from known classes with which to evaluate the model)
train_classes = 50
valid_classes = 15
test_classes = 15
total_classes = train_classes + valid_classes + test_classes

# calculate the class idxes to split on, use stratified sampling as the validation
# set is more difficult than the train set
train1, valid1, test1 = sample_data(train, 40, 12, 12)
train2, valid2, test2 = sample_data(valid, 10, 3, 3)
train_disjoint = append_json_data(train1, train2)
valid_disjoint = append_json_data(valid1, valid2)
test_disjoint = append_json_data(test1, test2)

# Append the data sets as we want to perform 80-way testing
valid_union = append_json_data(train_disjoint, valid_disjoint)
test_union = append_json_data(valid_union, test_disjoint)

# Check the correct number of classes is allocated
assert len(train_disjoint) == train_classes
assert len(valid_disjoint) == valid_classes
assert len(test_disjoint) == test_classes
assert len(valid_union) == train_classes + valid_classes
assert len(test_union) == total_classes

# Write to file, note we also want to perform the standard form of evaluation
# So we write the disjoint sets as well
write_json_file(train_disjoint, "train_disjoint.json")
write_json_file(valid_disjoint, "valid_disjoint.json")
write_json_file(test_disjoint, "test_disjoint.json")
write_json_file(valid_union, "valid_union.json")
write_json_file(test_union, "test_union.json")
