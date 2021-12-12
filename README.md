# The Applications of Meta-learners to Few-Shot Incremental Learning
Meta-learning is a fascinating field, meta-learners have the ability to generalise to predict classes they have never seen without any retraining, only 1 example of a new class is needed to classify into it. Meta-learners are explicitly trained for few-shot learning and yet their evaluation in literature is restricted to simulated problems where only a subset of classes are classified across. Additionally while they can classify given 5 examples, typically the need to be trained on a dataset with far more than 5 examples per class, so they don't align well with few-shot learning. 

In my thesis I argued that Incremental Learning presents a better application for Meta-learners, as this problem requires few-shot learning (of only new classes). Then I improved on the few incremental meta-learners available, finding that considering all the information in a Meta-Learning episode is essential.

This repository contains
- Thesis.pdf
- The code for the implementation of MetaFormer and other baselines
