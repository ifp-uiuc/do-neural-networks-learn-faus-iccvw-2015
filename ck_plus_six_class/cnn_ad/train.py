import argparse
import os
import sys
sys.path.append('..')

import numpy

from anna import util
from anna.datasets import supervised_dataset
#from anna.datasets.supervised_data_loader import SupervisedDataLoaderCrossVal

import data_fold_loader
import data_paths
from model import SupervisedModel


def reindex_labels(y):
    label_mapping = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    for label in label_mapping.keys():
        y[y == label] = label_mapping[label]

    return y


def add_padding(X, y):
    padding_shape = (64-X.shape[0], X.shape[1],
                     X.shape[2], X.shape[3])
    data_padding = numpy.zeros(padding_shape, dtype=X.dtype)
    label_padding = numpy.zeros(padding_shape[0], dtype=y.dtype)
    X_pad = numpy.concatenate((X, data_padding), axis=0)
    y_pad = numpy.concatenate((y, label_padding), axis=0)
    return X_pad, y_pad


parser = argparse.ArgumentParser(prog='train_cnn_with_dropout_\
                                 data_augmentation_six_class',
                                 description='Script to train convolutional \
                                 network from random initialization with \
                                 dropout and data augmentation \
                                 on six classes.')
parser.add_argument("-s", "--split", default='0', help='Testing split of CK+ \
                    to use. (0-9)')
parser.add_argument("--checkpoint_dir", default='./', help='Location to save \
                    model checkpoint files.')
args = parser.parse_args()


print('Start')
test_split = int(args.split)
if test_split < 0 or test_split > 9:
    raise Exception("Testing Split must be in range 0-9.")
print('Using CK+ testing split: {}'.format(test_split))

checkpoint_dir = os.path.join(args.checkpoint_dir, 'checkpoints_'+str(test_split))
print 'Checkpoint dir: ', checkpoint_dir

pid = os.getpid()
print('PID: {}'.format(pid))
f = open('pid_'+str(test_split), 'wb')
f.write(str(pid)+'\n')
f.close()

# Load model
model = SupervisedModel('experiment', './', learning_rate=1e-2)
monitor = util.Monitor(model,
                       checkpoint_directory=checkpoint_dir,
                       save_steps=1000)

# Add dropout to fully-connected layer
model.fc4.dropout = 0.5
model._compile()

# Loading CK+ dataset
print('Loading Data')
#supervised_data_loader = SupervisedDataLoaderCrossVal(
#    '/data/Expr_Recog/CK+_condensed/npy_files/')
#train_data_container = supervised_data_loader.load('train', train_split)
#test_data_container = supervised_data_loader.load('test', train_split)

train_folds, val_fold, _ = data_fold_loader.load_fold_assignment(test_fold=test_split)
X_train, y_train = data_fold_loader.load_folds(data_paths.ck_plus_data_path, train_folds)
X_val, y_val = data_fold_loader.load_folds(data_paths.ck_plus_data_path, [val_fold])
X_test, y_test = data_fold_loader.load_folds(data_paths.ck_plus_data_path, [test_split])

train_mask = numpy.logical_and(y_train != 0, y_train != 2)
X_train = X_train[train_mask, :, :, :]
y_train = y_train[train_mask]
y_train = reindex_labels(y_train)

val_mask = numpy.logical_and(y_val != 0, y_val != 2)
X_val = X_val[val_mask, :, :, :]
y_val = y_val[val_mask]
y_val = reindex_labels(y_val)

print 'Data train: ', X_train.shape, y_train.shape
print 'Data val: ', X_val.shape, y_val.shape

#test_mask = numpy.logical_and(y_test != 0, y_test != 2)
#X_test = X_test[test_mask, :, :, :]
#y_test = y_test[test_mask]
#y_test = reindex_labels(y_test)

#print ''
#print 'Train Data: ', X_train.shape, y_train.shape
#print 'Val Data: ', X_val.shape, y_val.shape
#print 'Test Data: ', X_test.shape, y_test.shape

if test_split == 8:
    # Val dataset only has 60 images (< 64 images = batch_size)
    X_val, y_val = add_padding(X_val, y_val)

X_train = numpy.float32(X_train)
X_train /= 255.0
X_train *= 2.0

X_val = numpy.float32(X_val)
X_val /= 255.0
X_val *= 2.0

train_dataset = supervised_dataset.SupervisedDataset(X_train, y_train)
val_dataset = supervised_dataset.SupervisedDataset(X_val, y_val)
train_iterator = train_dataset.iterator(
    mode='random_uniform', batch_size=64, num_batches=31000)
val_iterator = val_dataset.iterator(
    mode='random_uniform', batch_size=64, num_batches=31000)

# Do data augmentation (crops, flips, rotations, scales, intensity)
data_augmenter = util.DataAugmenter2(crop_shape=(96, 96),
                                     flip=True, gray_on=True)
normer = util.Normer3(filter_size=5, num_channels=1)
module_list_train = [data_augmenter, normer]
module_list_val = [normer]
preprocessor_train = util.Preprocessor(module_list_train)
preprocessor_val = util.Preprocessor(module_list_val)

print('Training Model')
for x_batch, y_batch in train_iterator:
    x_batch = preprocessor_train.run(x_batch)
    monitor.start()
    log_prob, accuracy = model.train(x_batch, y_batch)
    monitor.stop(1-accuracy)

    if monitor.test:
        monitor.start()
        x_val_batch, y_val_batch = val_iterator.next()
        x_val_batch = preprocessor_val.run(x_val_batch)
        val_accuracy = model.eval(x_val_batch, y_val_batch)
        monitor.stop_test(1-val_accuracy)
