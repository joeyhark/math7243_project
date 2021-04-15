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
from models.pointnet_sem_seg_model import pointnet_SEM_SEG_Model

DATASET = "new9"

VISUALIZE = False
#HAS TO BE TRUE FOR PNET 1
SAMPLE = True

tf.random.set_seed(42)


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


def load_test_dataset(in_file, batch_size):

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
	# dataset = dataset.batch(batch_size, drop_remainder=True)
	return dataset


def test(in_config=None):
	if in_config:
		config = in_config
	test_ds = load_test_dataset(config['test_ds'], config['batch_size'])

	# model = original_SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])
	# model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])
	# model = reduced2_SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])
  	#NOT FINISHED
	model = pointnet_SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])

	accuracies = []

	model.load_weights('./logs/{}/model/weights'.format(config['log_dir']))
	points = 8192
	for x in test_ds:
		data = x[0].numpy()
		truth_labels = x[1].numpy().flatten()
		# print(data.shape)

		if SAMPLE:
			indicies = sorted(random.sample(list(range(len(data))), k=points))
			sampled_data = data[indicies]
			sampled_labels = truth_labels[indicies]
		else:
			sampled_data = data
			sampled_labels = truth_labels

		sudo_batch = np.repeat([sampled_data], 4, axis=0)
		t1 = time.time()
		all_eval = model(sudo_batch)[0]
		# print(eval)
		print(f"model time: {time.time()-t1}")

		eval = np.argmax(all_eval, axis=1)

		#interpolate
		t2 = time.time()
		full_labels = interpolate_dense_labels(sampled_data, eval, data)
		# utils.simple_knn(sampled_data, eval, data)
		print(f'interpolate time: {time.time()-t2}')

		# print(eval.shape)
		wrong = sum(np.absolute(truth_labels - full_labels))
		acc = (len(data)-wrong)/len(data)
		print(f"test accuracy: {acc}")
		accuracies.append(acc)


		head = data[full_labels.astype(bool)]

		# distances = np.linalg.norm(head-np.mean(head, axis=0), axis=1)
		# reduced_head = head[np.where(distances < (np.mean(distances)+(1*np.std(distances))), True, False)]
		# print(f"reduced head len: {len(reduced_head)}")
		reduced_head = head


		# full_head = data[np.asarray(full_labels).astype(bool)]
		#
		if VISUALIZE:
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(data)
			o3d.visualization.draw_geometries([pcd])
			#
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(reduced_head)
			o3d.visualization.draw_geometries([pcd])


	print(f"Overall accuracy: {np.mean(accuracies)}")

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
    return np.array(dense_labels)

if __name__ == '__main__':

	config = {
		'train_ds' : f'data/{DATASET}/train/all.tfrecord',
		'val_ds' : f'data/{DATASET}/val/all.tfrecord',
		# 'test_ds' : f'data/{DATASET}/test.tfrecord',
		'test_ds' : f'data/{DATASET}/testBUT_TRAIN.tfrecord',
		'log_dir' : 'zimaging_pnet1_n9_1',
		'log_freq' : 10,
		'test_freq' : 100,
		'batch_size' : 4,
		'num_classes' : 2,
		'lr' : 0.001,
		'bn' : False,
	}

	test(config)
