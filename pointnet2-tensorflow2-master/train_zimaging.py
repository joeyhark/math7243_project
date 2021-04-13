import os
import sys
import datetime
import numpy as np
import open3d as o3d
import time
import random
from pnet2_layers import utils

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

from models.sem_seg_model import SEM_SEG_Model, original_SEM_SEG_Model, reduced2_SEM_SEG_Model

import test_zimaging

DATASET = "new4"

tf.random.set_seed(42)

# def parse_function(example_proto):
# 	feature_description = {
# 		'points': tf.io.FixedLenFeature([], tf.string, default_value=''),
# 		'value': tf.io.FixedLenFeature([], tf.int64, default_value=123),
# 		}
#
# 	parsed_1 = tf.io.parse_single_example(example_proto, feature_description)
# 	# parsed_1['data'] = tf.io.parse_tensor(parsed_1['data'], out_type=tf.double)
#
# 	return tf.reshape(tf.io.parse_tensor(parsed_1['data'], out_type=tf.double),
# 	 				  [-1,settings.DATA_SIZE]), tf.cast(tf.reshape(parsed_1['value'], [1]), tf.float32)

def load_dataset(in_file, batch_size):
	print(in_file)
	assert os.path.isfile(in_file), '[error] dataset path not found'

	n_points = 8192
	shuffle_buffer = 1000

	def _extract_fn(data_record):

		in_features = {
			'points': tf.io.FixedLenFeature([], tf.string),
			'label': tf.io.FixedLenFeature([], tf.string)
		}
		parsed_1 = tf.io.parse_single_example(data_record, in_features)
		return tf.reshape(tf.io.parse_tensor(parsed_1['points'], out_type=tf.double),
						  [-1,3]), tf.cast(tf.reshape(tf.io.parse_tensor(parsed_1['label'], out_type=tf.bool), [-1]), tf.int64)

	def _preprocess_fn(points, labels):

		# points = sample['points']
		# labels = sample['labels']

		points = tf.reshape(points, (n_points, 3))
		labels = tf.reshape(labels, (n_points, 1))

		shuffle_idx = tf.range(points.shape[0])
		shuffle_idx = tf.random.shuffle(shuffle_idx)
		points = tf.gather(points, shuffle_idx)
		labels = tf.gather(labels, shuffle_idx)

		return points, labels

	dataset = tf.data.TFRecordDataset(in_file)
	dataset = dataset.shuffle(shuffle_buffer)
	dataset = dataset.map(_extract_fn)
	dataset = dataset.map(_preprocess_fn)
	dataset = dataset.batch(batch_size, drop_remainder=True)

	return dataset


def load_test_dataset(in_file):

	assert os.path.isfile(in_file), '[error] dataset path not found'

	def _extract_fn(data_record):

		in_features = {
			'points': tf.io.FixedLenFeature([], tf.string),
			'label': tf.io.FixedLenFeature([], tf.string)
		}
		parsed_1 = tf.io.parse_single_example(data_record, in_features)
		return tf.reshape(tf.io.parse_tensor(parsed_1['points'], out_type=tf.double),
						  [-1,3]), tf.cast(tf.reshape(tf.io.parse_tensor(parsed_1['label'], out_type=tf.bool), [-1]), tf.int64)

	def _preprocess_fn(points, labels):

		# points = sample['points']
		# labels = sample['labels']

		points = tf.reshape(points, (-1, 3))
		labels = tf.reshape(labels, (-1, 1))

		return points, labels

	dataset = tf.data.TFRecordDataset(in_file)
	dataset = dataset.map(_extract_fn)
	dataset = dataset.map(_preprocess_fn)

	return dataset

def train():

	model = original_SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])
	# model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])
	# model = reduced2_SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])

	train_ds = load_dataset(config['train_ds'], config['batch_size'])
	val_ds = load_dataset(config['val_ds'], config['batch_size'])
	test_ds = load_test_dataset(config['test_ds'])
	# view_test_ds = load_test_dataset(config['view_ds'])

	# for x in test_ds:
	# 	print(x)
	callbacks = [
		keras.callbacks.TensorBoard(
			'./logs/{}'.format(config['log_dir']), update_freq=50),
		keras.callbacks.ModelCheckpoint(
			'./logs/{}/model/weights'.format(config['log_dir']), 'val_sparse_categorical_accuracy', save_best_only=True),
		tf.keras.callbacks.EarlyStopping(
			monitor='val_loss', restore_best_weights=True, patience=3)
	]

	model.build((config['batch_size'], 8192, 3))
	print(model.summary())

	model.compile(
		optimizer=keras.optimizers.Adam(config['lr']),
		loss=keras.losses.SparseCategoricalCrossentropy(),
		metrics=[keras.metrics.SparseCategoricalAccuracy()]
	)

	model.fit(
		train_ds,
		validation_data=val_ds,
		validation_steps=10,
		validation_freq=1,
		callbacks=callbacks,
		epochs=100,
		verbose=1
	)

def interpolate_dense_labels(sparse_points, sparse_labels, dense_points, k=1):
    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(sparse_points)
    sparse_pcd_tree = o3d.geometry.KDTreeFlann(sparse_pcd)

    dense_labels = []
    for dense_point in dense_points:
        _, sparse_indexes, _ = sparse_pcd_tree.search_knn_vector_3d(
            dense_point, k
        )
        knn_sparse_labels = sparse_labels[sparse_indexes]
        dense_label = np.bincount(knn_sparse_labels).argmax()
        dense_labels.append(dense_label)
    return dense_labels

if __name__ == '__main__':

	config = {
		'train_ds' : f'data/{DATASET}/train/all.tfrecord',
		'val_ds' : f'data/{DATASET}/val/all.tfrecord',
		# 'test_ds' : f'data/{DATASET}/test.tfrecord',
		'test_ds' : f'data/{DATASET}/testBUT_TRAIN.tfrecord',
		'log_dir' : 'zimaging_original_1',
		'log_freq' : 10,
		'test_freq' : 100,
		'batch_size' : 8,
		'num_classes' : 2,
		'lr' : 0.001,
		'bn' : False,
	}

	train()
	test_zimaging.test()
