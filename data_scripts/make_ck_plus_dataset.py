import argparse
import glob
import os
import shutil
import sys
import time

import cv2
import numpy
import skimage.color
import skimage.io
import skimage.transform

from ifp_toolbox.faces import FaceDetector


class CKPlusCondenser(object):
    def __init__(self, original_dataset_path, condensed_dataset_path):
        if os.path.exists(condensed_dataset_path):
            print 'Condensed Dataset detected.'
            print 'Removing it.'
            shutil.rmtree(condensed_dataset_path)
        print 'Copying original dataset to new condensed dataset path.'
        shutil.copytree(original_dataset_path, condensed_dataset_path)

        self.image_path = os.path.join(condensed_dataset_path,
                                       'cohn-kanade-images')
        self.label_path = os.path.join(condensed_dataset_path,
                                       'Emotion_labels')

    def run(self):
        print '\nCondensing CK+ Dataset: '
        self.condense_dataset()
        print '\nCondensed CK+ Dataset Statistics: '
        self.compute_dataset_statistics()

    def condense_dataset(self):
        # Get list of folders with no label file
        no_label_list = self.find_empty_folders(self.label_path)
        print '%d empty sequences to be removed.' % len(no_label_list)

        # Remove image sequence if label folder exists but is empty
        print '\nRemoving image sequence folders that have no label.'
        self.remove_image_sequences(self.image_path,
                                    self.label_path,
                                    no_label_list)

        # Remove empty folders in label directory
        print '\nRemoving empty label folders.'
        self.remove_folders_in_list(no_label_list)

        # Keep only the first and last three images in each sequence
        print '\nKeeping only the first and ' \
            'last three images in each sequence.'
        self.reduce_all_image_sequences(self.image_path)

    def find_empty_folders(self, label_path):
        folder_list = []
        for dirpath, dirs, files in os.walk(label_path):
            if not dirs and not files:
                folder_list.append(dirpath)
        return sorted(folder_list)

    def remove_image_sequences(self, image_path, label_path, no_label_list):
        mismatched_image_paths = self.find_image_label_mismatch(image_path,
                                                                label_path)
        self.remove_folders_in_list(mismatched_image_paths)

        # Gather folder extensions that have no label file
        folder_extension_list = []
        for folder_path in no_label_list:
            path_split_list = folder_path.split(os.sep)
            folder_extension = os.path.join(path_split_list[-2],
                                            path_split_list[-1])
            folder_extension_list.append(folder_extension)

        # Prepend the image_path to get the image sequence location
        image_sequence_path_list = [os.path.join(image_path, ext) for ext
                                    in folder_extension_list]

        # Remove image sequences in list
        self.remove_folders_in_list(image_sequence_path_list)

    def find_image_label_mismatch(self, image_path, label_path):
        mismatched_image_paths = []
        image_subj_list = sorted(os.listdir(image_path))

        for subj in image_subj_list:
            seq_list = sorted(os.listdir(os.path.join(image_path, subj)))
            for seq in seq_list:
                if seq == '.DS_Store':
                    os.remove(os.path.join(image_path, subj, seq))
                    continue

                seq_label_path = os.path.join(label_path, subj, seq)
                if not os.path.exists(seq_label_path):
                    seq_path = os.path.join(image_path, subj, seq)
                    mismatched_image_paths.append(seq_path)

        print 'There are %d mismatched files.' % len(mismatched_image_paths)
        return mismatched_image_paths

    def remove_folders_in_list(self, folder_list):
        #
        # Helper function to remove folders listed in folder_list
        #
        for i, folder_path in enumerate(folder_list):
            # print '%d: Removing --- %s' % (i, folder_path)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            else:
                print 'Folder %s does not exist' % (folder_path)
            time.sleep(0.1)

            parent_path, ext = os.path.split(folder_path)
            # Check if parent folder is empty
            if os.listdir(parent_path) == []:
                # If so, remove it
                # print 'Parent dir %s is empty' % parent_path
                shutil.rmtree(parent_path)
            elif os.listdir(parent_path) == ['.DS_Store']:
                # print '.DS_Store file is present. Removing...'
                os.remove(os.path.join(parent_path, '.DS_Store'))
                # print 'Now parent dir %s is empty' % parent_path
                shutil.rmtree(parent_path)

    def reduce_all_image_sequences(self, image_path):
        subj_folder_list = sorted(os.listdir(image_path))
        for subj_folder in subj_folder_list:
            subj_path = os.path.join(image_path, subj_folder)
            print 'Processing: ', subj_path
            seq_folder_list = sorted(os.listdir(subj_path))
            for seq_folder in seq_folder_list:
                seq_path = os.path.join(subj_path, seq_folder)
                if not os.path.isdir(seq_path):
                    continue

                self.reduce_single_sequence(seq_path)

    def reduce_single_sequence(self, path):
        file_list = sorted(os.listdir(path))
        for f in file_list:
            if f == '.DS_Store':
                # print 'Found it!'
                os.remove(os.path.join(path, f))

        if len(file_list) < 4:
            print 'Folder contains < 4 files. No reduction needed.'
            return

        remove_list = file_list[1:-3]
        for remove_file in remove_list:
            # print 'Remove ', remove_file
            os.remove(os.path.join(path, remove_file))

    def count_num_sequences(self, path):
        subj_folder_list = sorted(os.listdir(path))
        num_subj_total = len(subj_folder_list)

        num_seq_per_subj = []
        num_files_per_subj = []
        for folder in subj_folder_list:
            seq_list = os.listdir(os.path.join(path, folder))
            seq_list = [s for s in seq_list if s != '.DS_Store']
            num_sequences = len(seq_list)
            num_seq_per_subj.append(num_sequences)
            for seq in seq_list:
                num_files = len(os.listdir(os.path.join(path, folder, seq)))
                num_files_per_subj.append(num_files)

        num_seq_total = numpy.sum(num_seq_per_subj)
        return num_subj_total, num_seq_total

    def compute_dataset_statistics(self):
        num_subjects, num_sequences = self.count_num_sequences(self.image_path)
        print 'Total Number of Image Sequences: %d' % num_sequences

        _, num_label_sequences = self.count_num_sequences(self.label_path)
        print 'Total Number of Label Sequences: %d' % num_label_sequences

        # Number of sequences that have corresponding labels in Emotion_labels
        glob_label_path = os.path.join(self.label_path, '*/*/*.txt')
        num_label_files = len(glob.glob(glob_label_path))
        print 'Number of sequences with correponding label ' \
            '.txt file: %d' % num_label_files

        print 'Total Number of Subjects: %d' % num_subjects

        glob_image_path = os.path.join(self.image_path, '*/*/*.png')
        num_images_total = len(glob.glob(glob_image_path))
        print 'Number of image files: %d' % num_images_total


class CKPlusFaceCropper(object):
    def __init__(self, input_path):
        print '\nDetecting and Cropping Faces'
        self.input_path = input_path
        self.image_path = os.path.join(input_path, 'cohn-kanade-images')

    def run(self):
        self.crop_and_align_all_faces(self.image_path)

    def write_list_to_file(self, file_path, item_list):
        f = open(file_path, 'wb')
        for item in item_list:
            f.write(item+'\n')
        f.close()

    def crop_and_align_all_faces(self, path):
        output_img_size = (96, 96)
        missed_faces = []

        all_image_paths = sorted(glob.glob(os.path.join(path, '*/*/*.png')))
        # print all_image_paths[0:20]

        for image_file_path in all_image_paths:
            # print 'Detecting Face: %s' % image_file_path
            I, success_flag = self.process_single_image(
                                                   image_file_path,
                                                   output_img_size)
            if not success_flag:
                missed_faces.append(image_file_path)
            skimage.io.imsave(os.path.join(image_file_path), I)

        print 'Missed Faces: ', sorted(missed_faces)
        missed_faces_file_path = os.path.join(self.input_path,
                                              'missed_faces.txt')
        self.write_list_to_file(missed_faces_file_path, missed_faces)

    def process_single_image(self, image_file_path, output_img_size):
            # Read in the image
            I = skimage.io.imread(image_file_path)

            # If image was in color:
            if len(I.shape) == 3:
                I = skimage.color.rgb2gray(I)
                I *= 255
                I = I.astype('uint8')

            if len(I.shape) != 3:
                I = I[:, :, numpy.newaxis]

            # Detect face and crop it out
            I_crop, success_flag = self.detect_crop_face(I)

            # If face was successfully detected.
            # Align face in 96x96 image
            if success_flag:
                    I_out = I_crop
                    I_out = skimage.transform.resize(I_out, (96, 96))
            else:
                I_out = I_crop

            return I_out[:, :, numpy.newaxis], success_flag

    def detect_crop_face(self, I):
        success_flag = False
        face_detector = FaceDetector(scale_factor=1.3, min_neighbors=5,
                                     min_size_scalar=0.5, max_size_scalar=0.8)
        faces = face_detector.detect_faces(I)

        # If face was not detected:
        if len(faces) == 0:
            # Try with more lenient conditions
            face_detector = FaceDetector(scale_factor=1.3,
                                         min_neighbors=3,
                                         min_size_scalar=0.5,
                                         max_size_scalar=0.8)
            faces = face_detector.detect_faces(I)
            if len(faces) == 0:
                print 'Missed the face!'
                return I, success_flag

        success_flag = True
        I_crop = face_detector.crop_face_out(I, faces[0])
        return I_crop, success_flag


class CKPlusNumpyFileGenerator(object):
    def __init__(self, save_path):
        self.save_path = os.path.join(save_path, 'npy_files')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.image_path = os.path.join(save_path, 'cohn-kanade-images')
        self.label_path = os.path.join(save_path, 'Emotion_labels')

    def run(self):
        print '\nSaving CK+ images and labels to .npy files.'

        # Get number of images
        glob_image_path = os.path.join(self.image_path, '*/*/*.png')
        num_samples = len(glob.glob(glob_image_path))

        X, y, subjs = self.make_data_label_mats(self.image_path,
                                                self.label_path,
                                                num_samples)
        folds = self.make_folds(subjs)

        self.save_out_data(self.save_path, X, y, subjs, folds)

    def make_data_label_mats(self, all_images_path,
                             all_labels_path, num_samples):
        # Initialize the data of interest
        image_shape = (96, 96, 1)
        X = numpy.zeros((num_samples, image_shape[2],
                         image_shape[0], image_shape[1]), dtype='uint8')
        y = numpy.zeros((num_samples), dtype='int32')
        all_subjs = numpy.zeros((num_samples), dtype='int32')

        total_sample_count = 0
        subj_list = sorted(os.listdir(all_images_path))

        # For each subject folder:
        for i, subj in enumerate(subj_list):
            print 'Subject: %d - %s' % (i, subj)

            # For each individual sequence in the subject folder:
            seq_path = os.path.join(all_images_path, subj)
            seq_list = sorted(os.listdir(seq_path))
            for j, seq in enumerate(seq_list):
                # Get the images of the sequence and the emotion label
                images = self.read_images(all_images_path, subj, seq,
                                          image_shape)
                label = self.read_label(all_labels_path, subj, seq)
                label_vec = numpy.array([0, label, label, label])

                index_slice = slice(total_sample_count,
                                    total_sample_count+len(images))
                X[index_slice] = images
                y[index_slice] = label_vec
                all_subjs[index_slice] = i
                total_sample_count += len(images)

        return X, y, all_subjs

    def read_images(self, all_images_path, subj, seq, image_shape):
        image_file_path = os.path.join(all_images_path, subj, seq)
        image_files = sorted(os.listdir(image_file_path))
        num_images = len(image_files)

        images = numpy.zeros((num_images, image_shape[2],
                              image_shape[0], image_shape[1]))
        for i, image_file in enumerate(image_files):
            # print image_file
            I = skimage.io.imread(os.path.join(image_file_path, image_file))
            I = I[:, :, numpy.newaxis]
            images[i, :, :, :] = I.transpose(2, 0, 1)

        return images

    def read_label(self, all_labels_path, subj, seq):
        label_file_path = os.path.join(all_labels_path, subj, seq)
        label_file = os.listdir(label_file_path)[0]
        f = open(os.path.join(label_file_path, label_file))
        label = f.read()
        f.close()
        # print label
        label = int(float(label))

        return label

    def make_folds(self, subjs, num_folds=10):
        print '\nMaking the folds.'
        folds = numpy.zeros((subjs.shape), dtype='int32')
        num_subj = len(numpy.unique(subjs))

        for i in range(num_folds):
            subjs_in_fold = numpy.arange(i, num_subj, 10)
            print 'Subjs in fold %d: %s' % (i, subjs_in_fold)

            indices = numpy.hstack(
                [numpy.where(subjs == j)[0] for j in subjs_in_fold])
            folds[indices] = i

        print 'Number of samples/fold: %s' % numpy.histogram(folds,
                                                             bins=10)[0]

        return folds

    def save_out_data(self, path, X, y, subjs, folds):
        if not os.path.exists(path):
            os.makedirs(path)

        numpy.save(os.path.join(path, 'X.npy'), X)
        numpy.save(os.path.join(path, 'y.npy'), y)
        numpy.save(os.path.join(path, 'subjs.npy'), subjs)
        numpy.save(os.path.join(path, 'folds.npy'), folds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='make_ck_plus_dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Script to load, process, and split the Extended '
                    'Cohn-Kanade (CK+) Dataset.')
    parser.add_argument('-ip', '--input_path', dest='input_path',
                        help='Path specifying location of downloaded '
                             'CK+ files.')
    parser.add_argument('-sp', '--save_path', dest='save_path',
                        default='./CK_PLUS_HERE',
                        help='Path specifying where to save \
                              the pre-processed dataset and the \
                              output (.npy) files.')
    args = parser.parse_args()

    print('\n================================================================')
    print('                Extended Cohn-Kanade Dataset Manager              ')
    print('================================================================\n')

    input_path = args.input_path
    save_path = args.save_path

    # Condense CK+ dataset
    condenser = CKPlusCondenser(input_path, save_path)
    condenser.run()

    # Detect and crop faces
    face_cropper = CKPlusFaceCropper(save_path)
    face_cropper.run()

    # Save out CK+ .npy files
    numpy_file_generator = CKPlusNumpyFileGenerator(save_path)
    numpy_file_generator.run()

    print '\nSuccessfully pre-processed the Extended Cohn-Kanade Dataset!'
