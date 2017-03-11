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
        prog='ck_plus_single_checkpoint_evaluator',
        description='Script to evaluate the performance of a checkpoint \
                     on CK+.')
    parser.add_argument("-s", "--split", default='0',
                        help='Testing split of CK+ to use. (0-9)')
    parser.add_argument("checkpoint_file",
                        help='Path to a single model checkpoint (.pkl file).')
    args = parser.parse_args()

    checkpoint_file = args.checkpoint_file
    test_split = int(args.split)
    dataset_path = data_paths.ck_plus_data_path

    print 'Checkpoint: %s' % checkpoint_file
    print 'Testing on split %d\n' % test_split

    # Load model
    model = SupervisedModel('evaluation', './')

    # Load dataset
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

    val_evaluator = util.Evaluator(model, val_data_container,
                                   checkpoint_file, preprocessor)

    test_evaluator = util.Evaluator(model, test_data_container,
                                    checkpoint_file, preprocessor)

    # For the inputted checkpoint, compute the overall val accuracy
    print 'Checkpoint: %s' % os.path.split(checkpoint_file)[1]
    val_evaluator.set_checkpoint(checkpoint_file)
    val_accuracy = val_evaluator.run()
    print 'Val Accuracy: %f\n' % val_accuracy

    # For the inputted checkpoint, cmopute the overall test accuracy
    print 'Checkoint: %s' % os.path.split(checkpoint_file)[1]
    test_evaluator.set_checkpoint(checkpoint_file)
    test_accuracy = test_evaluator.run()
    print 'Test Accuracy: %f\n' % test_accuracy
