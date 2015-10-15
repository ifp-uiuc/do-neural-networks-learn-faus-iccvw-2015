import argparse
import os

import numpy
from scipy.io import loadmat


def load_data(TFD_dict):
    X = TFD_dict['images']
    folds = TFD_dict['folds']
    y_expression = TFD_dict['labs_ex']
    y_id = TFD_dict['labs_id']

    X = X[:, numpy.newaxis, :, :]

    # Remove samples without expression label
    y_expression = y_expression[y_expression != -1]
    y_expression -= 1
    # print y_expression.shape,  y_expression.min(), y_expression.max()

    # Remove samples without id label
    y_id = y_id[y_id != -1]
    y_id = reindex_labels(y_id)
    # print y_id.shape,  y_id.min(), y_id.max(), len(numpy.unique(y_id))

    return X, y_expression, y_id, folds


def reindex_labels(y):
    unique_classes = numpy.unique(y)
    num_classes = len(unique_classes)
    num_samples = len(y)

    y_reindex = numpy.zeros(num_samples)
    for i, c in enumerate(unique_classes):
        y_reindex[i] = numpy.where(unique_classes == y[i])[0]

    return y_reindex


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_dir_structure(path):
    make_dirs(os.path.join(path, 'TFD_48'))
    make_dirs(os.path.join(path, 'TFD_96'))


def save_out_unlabeled_data_to_npy_file(save_path, X):
    save_path = os.path.join(save_path, 'unlabeled')
    make_dirs(save_path)
    numpy.save(os.path.join(save_path, 'X.npy'), X)


def save_out_labeled_data_to_npy_files(save_path, X, y, folds):
    num_folds = folds.shape[1]
    for i in range(num_folds):
        inds = folds[:, i] != 0
        X_save = X[inds, :, :, :]
        fold_save = (folds[inds, i]-1)

        split_path = os.path.join(save_path, 'split_'+str(i))
        make_dirs(split_path)
        numpy.save(os.path.join(split_path, 'X.npy'), X_save)
        numpy.save(os.path.join(split_path, 'y.npy'), y)
        numpy.save(os.path.join(split_path, 'folds.npy'), fold_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='make_tfd_dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Script to load and split the Toronto Face Dataset (TFD).')
    parser.add_argument('-ip', '--input_path', dest='input_path',
                        help='Path specifying location of TFD (.mat) files.')
    parser.add_argument('-sp', '--save_path', dest='save_path',
                        default='./TFD_HERE',
                        help='Path specifying where to save \
                              the dataset (.npy) files.')
    args = parser.parse_args()

    print('\n================================================================')
    print('                        TFD Dataset Manager                       ')
    print('================================================================\n')

    input_path = args.input_path
    save_path = os.path.join(args.save_path, 'npy_files')

    make_dir_structure(save_path)
    save_path_48 = os.path.join(save_path, 'TFD_48')
    save_path_96 = os.path.join(save_path, 'TFD_96')

    # Load 48x48 and 96x96 image data
    print 'Loading the TFD dataset'
    TFD_48_dict = loadmat(os.path.join(input_path, 'TFD_48x48.mat'))
    TFD_96_dict = loadmat(os.path.join(input_path, 'TFD_96x96.mat'))
    TFD_info = loadmat(os.path.join(input_path, 'TFD_info.mat'))
    [X_48, y_expression_48, y_id_48, folds_48] = load_data(TFD_48_dict)
    [X_96, y_expression_96, y_id_96, folds_96] = load_data(TFD_96_dict)
    # print X_48.shape, X_96.shape

    # Extracting unlabeled data
    X_u_48 = X_48[folds_48[:, 0] == 0, :, :, :]
    X_u_96 = X_96[folds_96[:, 0] == 0, :, :, :]
    # print X_u_48.shape, X_u_96.shape

    # Save out the 98,058 unlabeled faces to disk
    print '\nSaving Unlabeled 48x48 images'
    save_out_unlabeled_data_to_npy_file(save_path_48, X_u_48)
    print 'Saving Unlabeled 96x96 images'
    save_out_unlabeled_data_to_npy_file(save_path_96, X_u_96)

    # Save each fold of the labeled data individually
    # Each split folder contains:
    #   X - 4178 images
    #   y - 4178 expression labels ranging from 0-6
    #   folds - 4178 fold labels (0-train, 1-dev, 2-test)
    print '\nSaving Labeled Splits of 48x48 images'
    save_out_labeled_data_to_npy_files(save_path_48, X_48,
                                       y_expression_48, folds_48)
    print 'Saving Labeled Splits of 96x96 images'
    save_out_labeled_data_to_npy_files(save_path_96, X_96,
                                       y_expression_96, folds_96)

    print '\nSuccessfully pre-processed the Toronto Face Dataset!'
