import argparse
import glob
import os
import sys

import numpy

from anna.datasets.supervised_data_loader import SupervisedDataContainer
from anna import util

import data_fold_loader
import data_paths
from model import SupervisedModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ck_plus_checkpoint_checker',
        description='Script to select best performing checkpoint on CK+.')
    parser.add_argument("-s", "--split", default='0',
                        help='Testing split of CK+ to use. (0-9)')
    parser.add_argument("checkpoint_dir",
                        help='Folder containing all .pkl checkpoint files.')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    test_split = int(args.split)
    dataset_path = data_paths.ck_plus_data_path

    print 'Checkpoint directory: %s' % checkpoint_dir
    print 'Testing on split %d\n' % test_split

    # Load model
    model = SupervisedModel('evaluation', './')

    # Load data
    train_folds, val_fold, _ = data_fold_loader.load_fold_assignment(test_fold=test_split)
    X_val, y_val = data_fold_loader.load_folds(data_paths.ck_plus_data_path, [val_fold])
    X_test, y_test = data_fold_loader.load_folds(data_paths.ck_plus_data_path, [test_split])
    print X_val.shape, y_val.shape
    print X_test.shape, y_test.shape

    X_val = numpy.float32(X_val)
    X_val /= 255.0
    X_val *= 2.0

    X_test = numpy.float32(X_test)
    X_test /= 255.0
    X_test *= 2.0

    val_data_container = SupervisedDataContainer(X_val, y_val)
    test_data_container = SupervisedDataContainer(X_test, y_test)

    # Construct evaluator
    preprocessor = [util.Normer3(filter_size=5, num_channels=1)]

    checkpoint_file_list = sorted(
        glob.glob(os.path.join(checkpoint_dir, '*.pkl')))
    val_evaluator = util.Evaluator(model, val_data_container,
                                   checkpoint_file_list[0], preprocessor)
    test_evaluator = util.Evaluator(model, test_data_container,
                                    checkpoint_file_list[0], preprocessor)

    # For each checkpoint, compute the overall val accuracy
    val_accuracies = []
    for checkpoint in checkpoint_file_list:
        print 'Checkpoint: %s' % os.path.split(checkpoint)[1]
        val_evaluator.set_checkpoint(checkpoint)
        val_accuracy = val_evaluator.run()
        print 'Val Accuracy: %f\n' % val_accuracy
        val_accuracies.append(val_accuracy)

    # Find checkpoint that produced the highest val accuracy
    max_val_accuracy = numpy.max(val_accuracies)
    max_index = numpy.argmax(val_accuracies)
    max_checkpoint = checkpoint_file_list[max_index]
    print 'Max Checkpoint: %s' % max_checkpoint
    print 'Max Val Accuracy: %f' % max_val_accuracy

    # Compute test accuracy of chosen checkpoint
    test_evaluator.set_checkpoint(max_checkpoint)
    test_accuracy = test_evaluator.run()
    print 'Test Accuracy: %f' % test_accuracy
