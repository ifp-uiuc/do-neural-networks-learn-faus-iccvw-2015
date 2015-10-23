import argparse
import glob
import os
import sys

import numpy

from anna.datasets.supervised_data_loader import SupervisedDataLoaderCrossVal
from anna import util

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
                        help='Training split of CK+ to use. (0-9)')
    parser.add_argument("checkpoint_file",
                        help='Path to a single model checkpoint (.pkl file).')
    args = parser.parse_args()

    checkpoint_file = args.checkpoint_file
    fold = int(args.split)
    dataset_path = data_paths.ck_plus_data_path

    print 'Checkpoint: %s' % checkpoint_file
    print 'Testing on split %d\n' % fold

    # Load model
    model = SupervisedModel('evaluation', './')

    # Load dataset
    supervised_data_loader = SupervisedDataLoaderCrossVal(dataset_path)
    test_data_container = supervised_data_loader.load(mode='test', fold=fold)
    test_data_container.X = numpy.float32(test_data_container.X)
    test_data_container.X /= 255.0
    test_data_container.X *= 2.0

    # Remove samples with neutral and contempt labels
    mask = numpy.logical_and(test_data_container.y != 0,
                             test_data_container.y != 2)
    test_data_container.X = test_data_container.X[mask, :, :, :]
    test_data_container.y = test_data_container.y[mask]
    test_data_container.y = reindex_labels(test_data_container.y)
    num_test_samples = len(test_data_container.y)

    if fold == 9:
        test_data_container.X, test_data_container.y = add_padding(
            test_data_container.X, test_data_container.y)

    # Construct evaluator
    preprocessor = [util.Normer3(filter_size=5, num_channels=1)]

    evaluator = util.Evaluator(model, test_data_container,
                               checkpoint_file, preprocessor)

    # For the inputted checkpoint, compute the overall test accuracy
    accuracies = []
    print 'Checkpoint: %s' % os.path.split(checkpoint_file)[1]
    evaluator.set_checkpoint(checkpoint_file)

    if fold != 9:
        accuracy = evaluator.run()
    else:
        predictions = evaluator._get_predictions()
        predictions = predictions[0:num_test_samples]
        true_labels = test_data_container.y[0:num_test_samples]

        accuracy = 100.0 * (1.0 * numpy.sum(
            predictions == true_labels) / len(true_labels))

    print 'Accuracy: %f\n' % accuracy
