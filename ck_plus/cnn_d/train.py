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


parser = argparse.ArgumentParser(prog='train_cnn_with_dropout',
                                 description='Script to train convolutional \
                                 neural network from random initialization \
                                 with dropout.')
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
#    data_paths.ck_plus_data_path)
#train_data_container = supervised_data_loader.load('train', train_split)
#test_data_container = supervised_data_loader.load('test', train_split)
train_folds, val_fold, _ = data_fold_loader.load_fold_assignment(test_fold=test_split)
X_train, y_train = data_fold_loader.load_folds(data_paths.ck_plus_data_path, train_folds)
X_val, y_val = data_fold_loader.load_folds(data_paths.ck_plus_data_path, [val_fold])
X_test, y_test = data_fold_loader.load_folds(data_paths.ck_plus_data_path, [test_split])

print X_train.shape, y_train.shape
print X_val.shape, y_val.shape
print X_test.shape, y_test.shape

X_train = numpy.float32(X_train)
X_train /= 255.0
X_train *= 2.0

X_val = numpy.float32(X_val)
X_val /= 255.0
X_val *= 2.0

X_test = numpy.float32(X_test)
X_test /= 255.0
X_test *= 2.0

train_dataset = supervised_dataset.SupervisedDataset(X_train, y_train)
val_dataset = supervised_dataset.SupervisedDataset(X_val, y_val)
train_iterator = train_dataset.iterator(
    mode='random_uniform', batch_size=64, num_batches=31000)
val_iterator = val_dataset.iterator(
    mode='random_uniform', batch_size=64, num_batches=31000)

# Create object to local contrast normalize a batch.
# Note: Every batch must be normalized before use.
normer = util.Normer3(filter_size=5, num_channels=1)
module_list = [normer]
preprocessor = util.Preprocessor(module_list)

print('Training Model')
for x_batch, y_batch in train_iterator:
    x_batch = preprocessor.run(x_batch)
    monitor.start()
    log_prob, accuracy = model.train(x_batch, y_batch)
    monitor.stop(1-accuracy)  # monitor takes error instead of accuracy

    if monitor.test:
        monitor.start()
        x_val_batch, y_val_batch = val_iterator.next()
        x_val_batch = preprocessor.run(x_val_batch)
        val_accuracy = model.eval(x_val_batch, y_val_batch)
        monitor.stop_test(1-val_accuracy)
