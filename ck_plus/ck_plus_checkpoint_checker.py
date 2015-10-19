import argparse
import glob
import os
import sys

import numpy

from anna.datasets.supervised_data_loader import SupervisedDataLoaderCrossVal
from anna import util

import data_paths
from model import SupervisedModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ck_plus_checkpoint_checker',
        description='Script to select best performing checkpoint on CK+.')
    parser.add_argument("-s", "--split", default='0',
                        help='Training split of CK+ to use. (0-9)')
    parser.add_argument("checkpoint_dir",
                        help='Folder containing all .pkl checkpoint files.')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    fold = int(args.split)
    dataset_path = data_paths.ck_plus_data_path

    print 'Checkpoint directory: %s' % checkpoint_dir

    # Load model
    model = SupervisedModel('evaluation', './')

    # Load data
    supervised_data_loader = SupervisedDataLoaderCrossVal(dataset_path)
    test_data_container = supervised_data_loader.load(mode='test', fold=fold)
    test_data_container.X = numpy.float32(test_data_container.X)
    test_data_container.X /= 255.0
    test_data_container.X *= 2.0

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
        accuracy = evaluator.run()
        print 'Accuracy: %f\n' % accuracy
        accuracies.append(accuracy)

    # Find checkpoint that produced the highest accuracy
    max_accuracy = numpy.max(accuracies)
    max_index = numpy.argmax(accuracies)
    max_checkpoint = checkpoint_file_list[max_index]
    print 'Max Checkpoint: %s' % max_checkpoint
    print 'Max Accuracy: %f' % max_accuracy
