import argparse
import os
import sys
sys.path.append('..')

import numpy

from anna import util
from anna.datasets import supervised_dataset
from anna.datasets.supervised_data_loader import SupervisedDataLoader

import data_paths
from model import SupervisedModel


parser = argparse.ArgumentParser(prog='train_cnn_with_dropout',
                                 description='Script to train convolutional\
                                  network from random initialization with \
                                  dropout.')
parser.add_argument("-s", "--split", default='0', help='Training split of TFD \
                                                        to use. (0-4)')
args = parser.parse_args()

print('Start')
train_split = int(args.split)
if train_split < 0 or train_split > 4:
    raise Exception("Training Split must be in range 0-4.")
print('Using TFD training split: {}'.format(train_split))

pid = os.getpid()
print('PID: {}'.format(pid))
f = open('pid_'+str(train_split), 'wb')
f.write(str(pid)+'\n')
f.close()

# Load model
model = SupervisedModel('experiment', './', learning_rate=1e-2)
monitor = util.Monitor(model,
                       checkpoint_directory='checkpoints_'+str(train_split),
                       save_steps=1000)

# Add dropout flag to fully-connected layer
model.fc4.dropout = 0.5
model._compile()

# Loading TFD dataset
print('Loading Data')
supervised_data_loader = SupervisedDataLoader(
    os.path.join(data_paths.tfd_data_path, 'npy_files/TFD_96/split_'+str(train_split))
train_data_container = supervised_data_loader.load(0)
val_data_container = supervised_data_loader.load(1)
test_data_container = supervised_data_loader.load(2)

X_train = train_data_container.X
y_train = train_data_container.y
X_val = val_data_container.X
y_val = val_data_container.y
X_test = test_data_container.X
y_test = test_data_container.y

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
test_dataset = supervised_dataset.SupervisedDataset(X_test, y_test)
train_iterator = train_dataset.iterator(
    mode='random_uniform', batch_size=64, num_batches=31000)
test_iterator = test_dataset.iterator(
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
    monitor.stop(1-accuracy)

    if monitor.test:
        monitor.start()
        x_test_batch, y_test_batch = test_iterator.next()
        x_test_batch = preprocessor.run(x_test_batch)
        test_accuracy = model.eval(x_test_batch, y_test_batch)
        monitor.stop_test(1-test_accuracy)
