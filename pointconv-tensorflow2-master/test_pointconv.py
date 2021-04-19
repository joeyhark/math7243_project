import os
import sys
import datetime
import numpy as np
import open3d as o3d
import time
import random

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

from model_scannet import PointConvModel
import sklearn
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

DATASET = "new9"

VISUALIZE = False

#sample down to 35000 points for time test
SUPERSAMPLE = True

SAMPLE = False
INTERPOLATE = False

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

	model = PointConvModel(config['batch_size'], config['bn'])
	# model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])
	# model = reduced2_SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])

	accuracies = []

	model.load_weights('./logs/{}/model/weights'.format(config['log_dir']))
	points = 8192
	points = 35000
	all_labels = []
	all_predictions = []
	mtimes = []
	itimes = []
	for x in test_ds:
		data = x[0].numpy()
		truth_labels = x[1].numpy().flatten()
		# print(data.shape)
		# print(truth_labels.shape)
		# print(data)
		# print(np.max(data))
		# print(np.min(data))
		# exit()
		if SUPERSAMPLE:
			indicies = sorted(random.sample(list(range(len(data))), k=35000))
			data = data[indicies]
			truth_labels = truth_labels[indicies]

		if SAMPLE:
			indicies = sorted(random.sample(list(range(len(data))), k=points))
			sampled_data = data[indicies]
			sampled_labels = truth_labels[indicies]
		else:
			sampled_data = data
			sampled_labels = truth_labels

		t1 = time.time()
		all_eval = model(tf.convert_to_tensor([sampled_data]))[0]
		# print(eval)
		mtime = time.time()-t1
		print(f"model time: {mtime}")
		mtimes.append(mtime)

		eval = np.argmax(all_eval, axis=1)

		if INTERPOLATE:
			#interpolate
			t2 = time.time()
			full_labels = interpolate_dense_labels(sampled_data, eval, data)
			# utils.simple_knn(sampled_data, eval, data)
			itime = time.time()-t2
			print(f'interpolate time: {itime}')
			itimes.append(itime)
			eval = full_labels
			sampled_data = data
			sampled_labels = truth_labels

		# print(eval.shape)
		wrong = sum(np.absolute(sampled_labels - eval))
		acc = (len(sampled_labels)-wrong)/len(sampled_labels)
		print(f"test accuracy: {acc}")
		accuracies.append(acc)
		# print(eval)
		# print(sum(eval))

		all_labels += truth_labels.tolist()
		all_predictions += eval.tolist()

		head = sampled_data[eval.astype(bool)]
		# print(head.shape)

		# print(f'len of head: {len(head)}')
		# distances = np.linalg.norm(head-np.mean(head, axis=0), axis=1)
		# reduced_head = head[np.where(distances < (np.mean(distances)+(1*np.std(distances))), True, False)]
		# print(f"reduced head len: {len(reduced_head)}")
		reduced_head = head



		if VISUALIZE:
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(data)
			o3d.visualization.draw_geometries([pcd])
			#
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(reduced_head)
			o3d.visualization.draw_geometries([pcd])



	print()
	print(f"Mean model time: {np.mean(mtimes)}")
	# print(f"Mean interpolate time: {np.mean(itimes)}")
	print(f"Overall accuracy: {np.mean(accuracies)}")
	print(f"Min accuracy: {min(accuracies)}")
	print(f"Max accuracy: {max(accuracies)}")
	c_matrix = tf.math.confusion_matrix(all_labels, all_predictions).numpy()
	print(c_matrix)
	df_cm = pd.DataFrame(c_matrix, index = ["Background", "Head"],
                  columns = ["Background", "Head"])
	# plt.figure(figsize = (10,7))
	# sn.heatmap(df_cm, annot=True)
	# plt.show()

	cf_matrix = c_matrix
	group_names = ['True Neg','False Pos','False Neg','True Pos']
	group_counts = ["{0:0.0f}".format(value) for value in
	                cf_matrix.flatten()]
	group_percentages = ["{0:.2%}".format(value) for value in
	                     cf_matrix.flatten()/np.sum(cf_matrix)]
	labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
	          zip(group_names,group_counts,group_percentages)]
	labels = np.asarray(labels).reshape(2,2)
	sn.heatmap(df_cm, annot=labels, fmt='', cmap='Blues')

	plt.hist(accuracies)
	plt.show()

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
	physical_devices = tf.config.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

	config = {
		'train_ds' : f'data/{DATASET}/train/all.tfrecord',
		'val_ds' : f'data/{DATASET}/val/all.tfrecord',
		# 'test_ds' : f'data/{DATASET}/test.tfrecord',
		'test_ds' : f'data/{DATASET}/testBUT_TRAIN.tfrecord',
		'log_dir' : 'zimaging_pconv_n9_1',
		'log_freq' : 10,
		'test_freq' : 100,
		'batch_size' : 4,
		'bn' : False,
	}


	test(config)
