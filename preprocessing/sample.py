import open3d as o3d
import os
import numpy as np
import h5py
import math
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from shutil import copy


DATASET = "new4"
TRAIN_PERCENT = 0.7

SUP_DIR = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname( __file__ ))))
DATA_DIR = os.path.abspath(os.path.join(SUP_DIR, 'data'))
DATA_SET_DIR = os.path.abspath(os.path.join(DATA_DIR, DATASET))
H5_DIR = os.path.join(DATA_SET_DIR, "h5_files")

TRAIN_DIR = os.path.abspath(os.path.join(DATA_SET_DIR, 'train'))
TRAIN_H5 = os.path.join(TRAIN_DIR, "h5_files")
TRAIN_DATASET = os.path.join(TRAIN_DIR, "dataset")

VAL_DIR = os.path.abspath(os.path.join(DATA_SET_DIR, 'val'))
VAL_H5 = os.path.join(VAL_DIR, "h5_files")
VAL_DATASET = os.path.join(VAL_DIR, "dataset")

# TEST_DIR = os.path.abspath(os.path.join(DATA_SET_DIR, 'val'))
# TEST_H5 = os.path.join(VAL_DIR, "h5_files")
# TEST_DATASET = os.path.join(VAL_DIR, "dataset")


def make_folders():
    try:
        os.mkdir(DATA_SET_DIR)
    except:
        pass
    try:
        os.mkdir(H5_DIR)
    except:
        pass

    try:
        os.mkdir(TRAIN_DIR)
    except:
        pass
    try:
        os.mkdir(TRAIN_H5)
    except:
        pass
    try:
        os.mkdir(TRAIN_DATASET)
    except:
        pass

    try:
        os.mkdir(VAL_DIR)
    except:
        pass
    try:
        os.mkdir(VAL_H5)
    except:
        pass
    try:
        os.mkdir(VAL_DATASET)
    except:
        pass

    # try:
    #     os.mkdir(TEST_DIR)
    # except:
    #     pass
    # try:
    #     os.mkdir(TESTL_H5)
    # except:
    #     pass
    # try:
    #     os.mkdir(TEST_DATASET)
    # except:
    #     pass

def split():

    categories = os.listdir(H5_DIR)
    for cat in categories:
        cat_h5 = os.path.join(H5_DIR, cat)
        cat_train_h5 = os.path.join(TRAIN_H5, cat)
        cat_train_dataset = os.path.join(TRAIN_DATASET, cat)
        cat_val_h5 = os.path.join(VAL_H5, cat)
        cat_val_dataset = os.path.join(VAL_DATASET, cat)

        try:
            os.mkdir(cat_train_h5)
        except:
            pass
        try:
            os.mkdir(cat_train_dataset)
        except:
            pass
        try:
            os.mkdir(cat_val_h5)
        except:
            pass
        try:
            os.mkdir(cat_val_dataset)
        except:
            pass

        scans = os.listdir(cat_h5)
        train_scans, val_scans = train_test_split(scans, train_size=float(TRAIN_PERCENT), shuffle=True)
        for scan in train_scans:
            copy(os.path.join(cat_h5, scan), os.path.join(cat_train_h5, scan))
        for scan in val_scans:
            copy(os.path.join(cat_h5, scan), os.path.join(cat_val_h5, scan))


def sample(dir):
    points = 8192
    samples_per_file = 100

    h5s = os.path.abspath(os.path.join(dir, 'h5_files'))
    dataset_dir = os.path.abspath(os.path.join(dir, "dataset"))
    all_tfrecord_file = os.path.abspath(os.path.join(dir, f'all.tfrecord'))

    all_sampled_data_list = []
    all_sampled_labels_list = []

    categories = os.listdir(h5s)
    for cat in categories:
        print(f"\nCategory: {cat}")
        cat_h5s = os.path.join(h5s, cat)
        tfrecord_file = os.path.abspath(os.path.join(dataset_dir, f'{cat}.tfrecord'))
        files = os.listdir(cat_h5s)
        sampled_data_list = []
        sampled_labels_list = []
        for file in files:
            print(f"Sampling {file}")
            with h5py.File(os.path.join(cat_h5s, file), "r") as h5_data:
                data = np.asarray(h5_data['data'])
                labels = np.asarray(h5_data['labels'])
                total_points = len(data)

                for i in range(samples_per_file):
                    indicies = sorted(random.sample(list(range(total_points)), k=points))
                    sampled_data = data[indicies]
                    sampled_labels = labels[indicies]
                    sampled_data_list.append(sampled_data)
                    sampled_labels_list.append(sampled_labels)

        # print(len(sampled_data_list))
        convert_to_tfrecord(sampled_data_list, sampled_labels_list, tfrecord_file)
        all_sampled_data_list += sampled_data_list
        all_sampled_labels_list += sampled_labels_list

    convert_to_tfrecord(all_sampled_data_list, all_sampled_labels_list, all_tfrecord_file)

# def sample_test():
#     points = 8192
#     samples_per_file = 100
#
#     raw_test_dir = os.path.abspath(os.path.join(TEST_DIR, 'raw_h5'))
#     raw_files = os.listdir(raw_test_dir)
#
#     sampled_data_list = []
#     sampled_labels_list = []
#     for file in raw_files:
#         with h5py.File(os.path.join(raw_test_dir, file), "r") as h5_data:
#             data = np.asarray(h5_data['data'])
#             labels = np.asarray(h5_data['labels'])
#             total_points = len(data)
#
#             for i in range(samples_per_file):
#                 indicies = sorted(random.sample(list(range(total_points)), k=points))
#                 sampled_data = data[indicies]
#                 sampled_labels = labels[indicies]
#                 sampled_data_list.append(sampled_data)
#                 sampled_labels_list.append(sampled_labels)
#     print(len(sampled_data_list))
#     tfrecord_file = os.path.abspath(os.path.join(TEST_DIR, 'sampled', f'test.tfrecord'))
#     convert_to_tfrecord(sampled_data_list, sampled_labels_list, tfrecord_file)

def convert_vis():
    tfrecord_file = os.path.abspath(os.path.join(DATA_SET_DIR, f'testBUT_TRAIN.tfrecord'))

    data_list = []
    labels_list = []

    categories = os.listdir(TRAIN_H5)
    for cat in categories:
        print(f"\nCategory: {cat}")
        cat_h5s = os.path.join(TRAIN_H5, cat)
        files = os.listdir(cat_h5s)
        for file in files:
            print(f"Sampling {file}")
            with h5py.File(os.path.join(cat_h5s, file), "r") as h5_data:
                data = np.asarray(h5_data['data'])
                labels = np.asarray(h5_data['labels'])
                data_list.append(data)
                labels_list.append(labels)

        # print(len(sampled_data_list))
    convert_to_tfrecord(data_list, labels_list, tfrecord_file)


def convert_to_tfrecord(data_list, labels_list, file):
    with tf.io.TFRecordWriter(str(file)) as writer:
        # for each example
        for i in range(len(data_list)):
            # create an item in the datset converted to the correct formats (float, int, byte)
            example = serialize_example(
                {
                    "points": {
                        "data": tf.io.serialize_tensor(data_list[i]),
                        "_type": _bytes_feature,
                    },

                    "label": {
                        "data": tf.io.serialize_tensor(labels_list[i]),
                        "_type": _bytes_feature,
                    },
                }
            )
            # write the defined example into the dataset
            # print('writing')
            writer.write(example)



def serialize_example(example):
    """Serialize an item in a dataset
    Arguments:
      example {[list]} -- list of dictionaries with fields "name" , "_type", and "data"

    Returns:
      [type] -- [description]
    """
    dset_item = {}
    for feature in example.keys():
        dset_item[feature] = example[feature]["_type"](example[feature]["data"])
        example_proto = tf.train.Example(features=tf.train.Features(feature=dset_item))
    return example_proto.SerializeToString()


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




if __name__ == '__main__':
    # make_folders()
    # split()
    # sample(TRAIN_DIR)
    # sample(VAL_DIR)
    convert_vis()
