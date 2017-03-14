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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ck_plus_single_checkpoint_evaluator',
        description='Script to evaluate the performance of a checkpoint \
                     on CK+ (six classes).')
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
    #supervised_data_loader = SupervisedDataLoaderCrossVal(dataset_path)
    #test_data_container = supervised_data_loader.load(mode='test', fold=fold)
    #test_data_container.X = numpy.float32(test_data_container.X)
    #test_data_container.X /= 255.0
    #test_data_container.X *= 2.0

    train_folds, val_fold, _ = data_fold_loader.load_fold_assignment(test_fold=test_split)
    X_val, y_val = data_fold_loader.load_folds(data_paths.ck_plus_data_path, [val_fold])
    X_test, y_test = data_fold_loader.load_folds(data_paths.ck_plus_data_path, [test_split])
    print 'Val Data: ', X_val.shape, y_val.shape
    print 'Test Data: ', X_test.shape, y_test.shape

    X_val = numpy.float32(X_val)
    X_val /= 255.0
    X_val *= 2.0

    X_test = numpy.float32(X_test)
    X_test /= 255.0
    X_test *= 2.0

    val_data_container = SupervisedDataContainer(X_val, y_val)
    test_data_container = SupervisedDataContainer(X_test, y_test)


    # Remove samples with neutral and contempt labels
    val_mask = numpy.logical_and(y_val != 0, y_val != 2)
    X_val = X_val[val_mask, :, :, :]
    y_val = y_val[val_mask]
    y_val = reindex_labels(y_val)
    num_val_samples = len(y_val)

    mask_test = numpy.logical_and(y_test != 0, y_test != 2)
    X_test = X_test[mask_test, :, :, :]
    y_test = y_test[mask_test]
    y_test = reindex_labels(y_test)
    num_test_samples = len(y_test)

    print 'Reduced Val Data: ', X_val.shape, y_val.shape
    print 'Reduced Test Data: ', X_test.shape, y_test.shape

    if test_split == 9:
        X_test, y_test = add_padding(X_test, y_test)
    elif test_split == 8:
        X_val, y_val = add_padding(X_val, y_val)

    val_data_container = SupervisedDataContainer(X_val, y_val)
    test_data_container = SupervisedDataContainer(X_test, y_test)

    # Construct evaluator
    preprocessor = [util.Normer3(filter_size=5, num_channels=1)]

    val_evaluator = util.Evaluator(model, val_data_container,
                                   checkpoint_file, preprocessor)
    test_evaluator = util.Evaluator(model, test_data_container,
                                    checkpoint_file, preprocessor)

    # For the inputted checkpoint, compute the overall test accuracy
    #accuracies = []
    print 'Checkpoint: %s' % os.path.split(checkpoint_file)[1]
    val_evaluator.set_checkpoint(checkpoint_file)
 
    if test_split != 8:
        val_accuracy = val_evaluator.run()
    else:
        val_predictions = val_evaluator._get_predictions()
        val_predictions = val_predictions[0:num_val_samples]
        val_true_labels = val_data_container.y[0:num_val_samples]

        val_accuracy = 100.0 * (1.0 * numpy.sum(
            val_predictions == val_true_labels) / len(val_true_labels))

    print 'Val Accuracy: %f\n' % val_accuracy


    print 'Checkpoint: %s' % os.path.split(checkpoint_file)[1]
    test_evaluator.set_checkpoint(checkpoint_file)

    if test_split != 9:
        test_accuracy = test_evaluator.run()
    else:
        test_predictions = test_evaluator._get_predictions()
        test_predictions = test_predictions[0:num_test_samples]
        test_true_labels = test_data_container.y[0:num_test_samples]

        test_accuracy = 100.0 * (1.0 * numpy.sum(
            test_predictions == test_true_labels) / len(test_true_labels))

    print 'Test Accuracy: %f\n' % test_accuracy

