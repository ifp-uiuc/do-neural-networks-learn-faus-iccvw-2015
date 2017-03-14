import os
import numpy


def load_folds(dataset_path, fold_nums):
    X = numpy.load(os.path.join(dataset_path, 'X.npy'))
    y = numpy.load(os.path.join(dataset_path, 'y.npy'))
    folds = numpy.load(os.path.join(dataset_path, 'folds.npy'))

    #assert fold <= folds.max(), \
    #    'Fold number exceeds available number of folds. Please try again.'

    X_all = []
    y_all = []
    for fold in fold_nums:
        mask = (folds == fold)

	print 'Fold %d' % fold
	X_fold = X[mask, :, :, :]
	y_fold = y[mask]
	print X_fold.shape, y_fold.shape

        X_all.append(X_fold)
        y_all.append(y_fold)

    X_all = numpy.concatenate(X_all, axis=0)
    y_all = numpy.concatenate(y_all, axis=0)
    print 'X_all shape: ', X_all.shape
    print 'y_all shape: ', y_all.shape

    return X_all, y_all


def load_fold_assignment(test_fold):
    val_fold = (test_fold + 1) % 10
    train_fold = list(set(range(0, 10)) - set([test_fold]) - set([val_fold]))

    print 'Test_fold: ', test_fold
    print 'Val_fold: ', val_fold
    print 'Train_folds: ', train_fold

    return train_fold, val_fold, test_fold
