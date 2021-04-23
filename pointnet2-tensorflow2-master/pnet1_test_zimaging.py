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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.sem_seg_model import SEM_SEG_Model, original_SEM_SEG_Model, reduced2_SEM_SEG_Model
from models.pointnet_sem_seg_model import pointnet_SEM_SEG_Model
import sklearn
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

DATASET = "new9"

VISUALIZE = False
MIN_VIS = 1
MAX_VIS = 0.99
#sample down to 35000 points for time test
SUPERSAMPLE = True
#stack if not sample
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
	all_labels = []
	all_predictions = []
	mtimes = []
	itimes = []
	atimes = []
	for x in test_ds:
		t0 = time.time()
		data = x[0].numpy()
		truth_labels = x[1].numpy().flatten()
		# print(data.shape)
		if SUPERSAMPLE:
			indicies = sorted(random.sample(list(range(len(data))), k=35000))
			data = data[indicies]
			truth_labels = truth_labels[indicies]



		if SAMPLE:
			indicies = sorted(random.sample(list(range(len(data))), k=points))
			sampled_data = data[indicies]
			sampled_labels = truth_labels[indicies]

			sudo_batch = np.repeat([sampled_data], 4, axis=0)
			t1 = time.time()
			all_eval = model(sudo_batch)[0]
			# print(eval)
			mtime = time.time()-t1
			print(f"model time: {mtime}")
			mtimes.append(mtime)
			eval = np.argmax(all_eval, axis=1)

			t2 = time.time()
			full_labels = interpolate_dense_labels(sampled_data, eval, data)
			# utils.simple_knn(sampled_data, eval, data)
			itime = time.time()-t2
			print(f'interpolate time: {itime}')
			itimes.append(itime)

		else:
			#stack
			z = list(zip(data, truth_labels))
			random.shuffle(z)
			data, truth_labels = zip(*z)
			truth_labels = np.array(truth_labels)

			stack = []
			n = len(data)
			four_bin = 8192*4
			if n % four_bin != 0:
				bins = n//four_bin + 1
				data_expanded = data + data[:four_bin-(n-((bins-1)*four_bin))]
			else:
				bins = n//four_bin
				data_expanded = data
			data = np.array(data)
			data_binned = np.reshape(data_expanded, [-1,4,8192,3])
			# print(data_binned.shape)
			all_evals = []
			mtime_total = 0
			for data_bin in data_binned:
				data_bin = tf.convert_to_tensor(data_bin)
				# print(data_bin.shape)
				t1 = time.time()
				bin_eval = model(data_bin).numpy()
				mtime = time.time()-t1
				mtime_total += mtime

				# print(bin_eval.shape)
				bin_eval = np.argmax(np.reshape(bin_eval, [-1,2]), axis=1)
				# print(bin_eval.shape)
				all_evals += list(bin_eval)
			full_labels = np.array(all_evals).flatten()[:n]
			# print(full_labels)
			print(f"model time: {mtime_total}")
			mtimes.append(mtime_total)

		atime = time.time()-t0
		print(f"all time: {atime}")
		atimes.append(atime)
		#interpolate



		# print(eval.shape)
		wrong = sum(np.absolute(truth_labels - full_labels))
		acc = (len(data)-wrong)/len(data)
		print(f"test accuracy: {acc}")
		accuracies.append(acc)

		all_labels += truth_labels.tolist()
		all_predictions += full_labels.tolist()

		head = data[full_labels.astype(bool)]

		# distances = np.linalg.norm(head-np.mean(head, axis=0), axis=1)
		# reduced_head = head[np.where(distances < (np.mean(distances)+(1*np.std(distances))), True, False)]
		# print(f"reduced head len: {len(reduced_head)}")
		reduced_head = head


		# full_head = data[np.asarray(full_labels).astype(bool)]
		#
		if VISUALIZE and acc < MIN_VIS and acc >= MAX_VIS:
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(data)
			o3d.visualization.draw_geometries([pcd])
			#
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(reduced_head)
			o3d.visualization.draw_geometries([pcd])


	print()
	print(f"Mean model time: {np.mean(mtimes)}")
	print(f"Mean all time: {np.mean(atimes)}")
	if SAMPLE:
		print(f"Mean interpolate time: {np.mean(itimes)}")
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

	config = {
		'train_ds' : f'data/{DATASET}/train/all.tfrecord',
		'val_ds' : f'data/{DATASET}/val/all.tfrecord',
		# 'test_ds' : f'data/{DATASET}/test.tfrecord',
		'test_ds' : f'data/{DATASET}/testBUT_TRAIN.tfrecord',
		# 'log_dir' : 'zimaging_pnet1_n9_lr=0.0001',
		'log_dir' : 'zimaging_pnet1_n9_lr=0.0005',
		'log_freq' : 10,
		'test_freq' : 100,
		'batch_size' : 4,
		'num_classes' : 2,
		'lr' : 0.001,
		'bn' : False,
	}

	test(config)
