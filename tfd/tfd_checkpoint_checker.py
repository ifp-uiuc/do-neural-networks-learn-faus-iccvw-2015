import argparse
import glob
import os
import sys

import numpy

from anna.datasets.supervised_data_loader import SupervisedDataLoader
from anna import util

import data_paths
from model import SupervisedModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='tfd_plus_checkpoint_checker',
        description='Script to select best performing checkpoint on TFD.')
    parser.add_argument("-s", "--split", default='0',
                        help='Training split of TFD to use. (0-4)')
    parser.add_argument("checkpoint_dir",
                        help='Folder containing all .pkl checkpoint files.')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    fold = int(args.split)
    dataset_path = os.path.join(data_paths.tfd_data_path, 'npy_files/TFD_96/split_'+str(fold))

    print 'Checkpoint directory: %s' % checkpoint_dir
    print 'Testing on split %d\n' % fold

    # Load model
    model = SupervisedModel('evaluation', './')

    # Load data
    supervised_data_loader = SupervisedDataLoader(dataset_path)
    test_data_container = supervised_data_loader.load(2)
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
