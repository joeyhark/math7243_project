import open3d as o3d
import os
import numpy as np
import h5py
import math
import random
import tensorflow as tf

SUP_DIR = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname( __file__ ))))
DATA_FOLDER = os.path.abspath(os.path.join(SUP_DIR, 'data'))
TRAIN_DIR = os.path.abspath(os.path.join(DATA_FOLDER, 'train'))
TEST_DIR = os.path.abspath(os.path.join(DATA_FOLDER, 'test'))


def sample():
    points = 8192
    samples_per_file = 100

    raw_train_dir = os.path.abspath(os.path.join(TRAIN_DIR, 'raw_h5'))
    raw_files = os.listdir(raw_train_dir)

    sampled_data_list = []
    sampled_labels_list = []
    for file in raw_files:
        with h5py.File(os.path.join(raw_train_dir, file), "r") as h5_data:
            data = np.asarray(h5_data['data'])
            labels = np.asarray(h5_data['labels'])
            total_points = len(data)

            for i in range(samples_per_file):
                indicies = sorted(random.sample(list(range(total_points)), k=points))
                sampled_data = data[indicies]
                sampled_labels = labels[indicies]
                sampled_data_list.append(sampled_data)
                sampled_labels_list.append(sampled_labels)
    print(len(sampled_data_list))
    tfrecord_file = os.path.abspath(os.path.join(TRAIN_DIR, 'sampled', f'train.tfrecord'))
    convert_to_tfrecord(sampled_data_list, sampled_labels_list, tfrecord_file)

def sample_test():
    points = 8192
    samples_per_file = 100

    raw_test_dir = os.path.abspath(os.path.join(TEST_DIR, 'raw_h5'))
    raw_files = os.listdir(raw_test_dir)

    sampled_data_list = []
    sampled_labels_list = []
    for file in raw_files:
        with h5py.File(os.path.join(raw_test_dir, file), "r") as h5_data:
            data = np.asarray(h5_data['data'])
            labels = np.asarray(h5_data['labels'])
            total_points = len(data)

            for i in range(samples_per_file):
                indicies = sorted(random.sample(list(range(total_points)), k=points))
                sampled_data = data[indicies]
                sampled_labels = labels[indicies]
                sampled_data_list.append(sampled_data)
                sampled_labels_list.append(sampled_labels)
    print(len(sampled_data_list))
    tfrecord_file = os.path.abspath(os.path.join(TEST_DIR, 'sampled', f'test.tfrecord'))
    convert_to_tfrecord(sampled_data_list, sampled_labels_list, tfrecord_file)

def convert_test():
    raw_test_dir = os.path.abspath(os.path.join(TEST_DIR, 'raw_h5'))
    raw_files = os.listdir(raw_test_dir)
    with h5py.File(os.path.join(raw_test_dir, raw_files[-1]), "r") as h5_data:
        data = np.asarray(h5_data['data'])
        labels = np.asarray(h5_data['labels'])
        tfrecord_file = os.path.abspath(os.path.join(TEST_DIR, 'sampled', f'test_full.tfrecord'))
        convert_to_tfrecord([data], [labels], tfrecord_file)

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
    # sample()
    # sample_test()
    convert_test()
