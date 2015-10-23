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
        prog='ck_plus_checkpoint_checker',
        description='Script to select best performing checkpoint \
                    on CK+ (six classes).')
    parser.add_argument("-s", "--split", default='0',
                        help='Training split of CK+ to use. (0-9)')
    parser.add_argument("checkpoint_dir",
                        help='Folder containing all .pkl checkpoint files.')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    fold = int(args.split)
    dataset_path = data_paths.ck_plus_data_path

    print 'Checkpoint directory: %s' % checkpoint_dir
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

    checkpoint_file_list = sorted(
        glob.glob(os.path.join(checkpoint_dir, '*.pkl')))
    evaluator = util.Evaluator(model, test_data_container,
                               checkpoint_file_list[0], preprocessor)

    # For each checkpoint, compute the overall test accuracy
    accuracies = []
    for checkpoint in checkpoint_file_list:
        print 'Checkpoint: %s' % os.path.split(checkpoint)[1]
        evaluator.set_checkpoint(checkpoint)

        if fold != 9:
            accuracy = evaluator.run()
        else:
            predictions = evaluator._get_predictions()
            predictions = predictions[0:num_test_samples]
            true_labels = test_data_container.y[0:num_test_samples]

            accuracy = 100.0 * (1.0 * numpy.sum(
                predictions == true_labels) / len(true_labels))

        print 'Accuracy: %f\n' % accuracy
        accuracies.append(accuracy)

    # Find checkpoint that produced the highest accuracy
    max_accuracy = numpy.max(accuracies)
    max_index = numpy.argmax(accuracies)
    max_checkpoint = checkpoint_file_list[max_index]
    print 'Max Checkpoint: %s' % max_checkpoint
    print 'Max Accuracy: %f' % max_accuracy
