# Running the CK+ experiments (six classes)

# Contents
+ [Introduction](#introduction)
+ [Folder contents](#folder-contents)
+ [Running experiments](#running-experiments)

# Introduction

This folder contains some of the code used to obtain our results on the [CK+][CK+] dataset. However, instead of training and evaluating on all eight expression classes, this model only deals with the **six** basic emotions: (anger, disgust, fear, happy, sad, and surprise). Our experiment uses both data augmentation and dropout as forms of regularization.

We will first describe the contents of this folder, and then walk you through
how to run the experiments.

## Folder contents
The folder contains:
``` shell
/cnn_ad/train.py
ck_plus_checkpoint_checker.py
ck_plus_single_checkpoint_evaluator.py
data_paths.py
model.py
```

#### `train.py`
Our train.py trains the CNN model specified in the ``model.py`` file. It also outputs a 
directory of model checkpoint files, and a log of the training process.

#### `ck_plus_single_checkpoint_evaluator.py`
This file contains a script that outputs the performance of a single save checkpoint.

#### `ck_plus_checkpoint_checker.py`
This file contains a script that examines all the checkpoints created by a 
single experiment, and chooses the best one.

#### `data_paths.py`
This file contains a single variable (``ck_plus_six_class_data_path``) which indicates the path to load the CK+ ``.npy`` files 
created when running the ``make_ck_plus_dataset.py`` file.

#### `model.py`
This contains the CNN model used that is loaded by the ``train.py`` file.


# Running experiments

## CNN Training

You are now ready the CNN.  

You can train the cnn with following command: 
``` shell
# Snippet: cnn training - six class
$ THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' \ 
python -u train.py --split 0  \ 
>& log0.txt & 
```

Since the [CK+][CK+] dataset is typically broken into splits and their results averaged,
the `--split` option indicates which of the 10 splits to use (0-9) when training. The code 
will save the `.pkl` file containing the network parameters to a directory called `./checkpoints_0/` 
which will denote the split used.

### How to evaluate a model's performance

After you have trained a split to completion, you can find the best performing
checkpoint by running the checkpoint evaluator found in 
`ck_plus_checkpoint_checker.py`. Simply run the following command:

``` shell
# Snippet: cnn checkpoint evaluation
$ THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' \ 
python -u ck_plus_checkpoint_checker.py --split 0 ./cnn_ad/checkpoints_0/ \
>& cnn_ad_best_performance_split_0.txt &
```

With this command, `ck_plus_checkpoint_checker.py` will iterate over the list of
checkpoints found in `./cnn_ad/checkpoints_0/` and compute the accuracy on 
the test set. It will then select the checkpoint that yielded the highest
accuracy. The command also writes all of the results to a text file called 
`cnn_ad_best_performance_split_0.txt`. 


[CK+]:http://www.pitt.edu/~emotion/ck-spread.htm
