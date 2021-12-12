Data retrieved from here https://github.com/betterAndTogether/IncreProtoNet

Note that the dataset naming scheme is confusing. 

TLDR: Validation data is "val_novel_fewrel.json", Test data is "novel_test_fewrel.json"
It appears that "novelrel2index.json" contains the validation classes, 
as it has 10 classes and keys in "novelrel2index_val.json" overlap with the base classes
NOTE: novel_val_fewrel uses "novelrel2index_val.json" wherease val_novel_fewrel uses the correct set

The test set classes are not outlined at all, so we should take the remaining classes as the 16 test set.

While the test set classes are not outlined in an index, the "novel_test_fewrel.json" file does
contain 16 classes that don't overlap with the base classes, so we can use that