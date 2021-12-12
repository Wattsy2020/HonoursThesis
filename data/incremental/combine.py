# A small script to combine the base_[val/test]_fewrel with novel_[val/test]_fewrel to create
# The same type of dataset used by the framework
import json

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

base_train = read_json_file("Original/base_train_fewrel.json")
base_val = read_json_file("Original/base_val_fewrel.json")
base_test = read_json_file("Original/base_test_fewrel.json")
novel_val = read_json_file("Original/val_novel_fewrel.json")
novel_test = read_json_file("Original/novel_test_fewrel.json")

val = append_json_data(base_val, novel_val)
test = append_json_data(base_test, novel_test)

write_json_file(base_train, "train.json")
write_json_file(val, "val.json")
write_json_file(test, "test.json")
