import os
import sys

sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, MaxPool2D

from pnet2_layers.layers import Pointnet_SA, Pointnet_FP
from pnet2_layers.utils import Conv2d

#hardcoded for now
NUM_POINTS = 8192

class pointnet_SEM_SEG_Model(Model):

	def __init__(self, batch_size, num_classes, bn=False, activation=tf.nn.relu):
		super(pointnet_SEM_SEG_Model, self).__init__()

		self.activation = activation
		self.batch_size = batch_size
		self.keep_prob = 0.5
		self.num_classes = num_classes
		self.bn = bn

		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.init_network()


	def init_network(self):
		self.conv1 = Conv2d(64, strides=[1,1], activation=self.activation, padding='VALID',
						 bn=self.bn)
		self.conv2 = Conv2d(64, strides=[1,1], activation=self.activation, padding='VALID',
                         bn=self.bn)
		self.conv3 = Conv2d(64, strides=[1,1], activation=self.activation, padding='VALID',
                         bn=self.bn)
		self.conv4 = Conv2d(128, strides=[1,1], activation=self.activation, padding='VALID',
                         bn=self.bn)
		self.conv5 = Conv2d(1024, strides=[1,1], activation=self.activation, padding='VALID',
                         bn=self.bn)

		self.pool1 = MaxPool2D(pool_size=[NUM_POINTS,1], padding='valid')

		# pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])

		self.dense1 = Dense(256, activation=self.activation)
		self.dense2 = Dense(128, activation=self.activation)

		# 	pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
		# points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])

		self.conv6 = Conv2d(512, strides=[1,1], activation=self.activation, padding='VALID')
		self.conv7 = Conv2d(256, strides=[1,1], activation=self.activation, padding='VALID')
		# self.dropout1 = Dropout(self.keep_prob)
		# self.conv8 = Conv2d(self.num_classes, strides=[1,1], activation=tf.nn.softmax, padding='VALID')

		# net = tf.squeeze(net, [2])

		self.dense3 = Dense(1024, activation=self.activation)
		self.dropout2 = Dropout(self.keep_prob)
		self.dense4 = Dense(self.num_classes, activation=tf.nn.softmax)

	def forward_pass(self, input, training):
		input_image = tf.expand_dims(input, -1)
		# print(f"input_image: {input_image.shape}")
		net = self.conv1(input_image, training=training)
		# print(f"conv1: {net.shape}")
		net = self.conv2(net, training=training)
		# print(f"conv2: {net.shape}")
		net = self.conv3(net, training=training)
		# print(f"conv3: {net.shape}")
		net = self.conv4(net, training=training)
		# print(f"conv4: {net.shape}")
		net_points = self.conv5(net, training=training)
		# print(f"conv5: {net_points.shape}")

		net = self.pool1(net_points, training=training)
		# print(f"pool1 net: {net.shape}")

		net = tf.reshape(net, [self.batch_size, -1])
		# print(f"reshape net: {net.shape}")

		net = self.dense1(net, training=training)
		# print(f"dense1: {net.shape}")
		net = self.dense2(net, training=training)
		# print(f"dense2: {net.shape}")

		net_expanded = tf.tile(tf.reshape(net, [self.batch_size, 1, 1, -1]), [1, NUM_POINTS, 3, 1])
		# print(f"tile (expanded): {net_expanded.shape}")
		net = tf.concat(values=[net_points, net_expanded], axis=3)
		# print(f"concat: {net.shape}")


		net = self.conv6(net, training=training)
		# print(f"conv6: {net.shape}")
		net = self.conv7(net, training=training)
		# print(f"conv7: {net.shape}")
		# net = self.dropout1(net, training=training)
		# print(f"dropout1: {net.shape}")

		# pred = self.conv8(net, training=training)
		# print(f"conv8/pred: {pred.shape}")

		"""MY MODIFICATIONS"""
		net = tf.reshape(net, [self.batch_size, NUM_POINTS, -1])
		# print(f"reshape net: {net.shape}")

		net = self.dense3(net, training=training)
		# print(f"dense3: {net.shape}")
		net = self.dropout2(net, training=training)
		# print(f"dropout2: {net.shape}")
		net = self.dense4(net, training=training)
		# print(f"dense4: {net.shape}")


		# net = tf.squeeze(net, [2])
		# print(f"squeeze: {net.shape}")

		return net


	def train_step(self, input):

		with tf.GradientTape() as tape:
			# print("\na")
			pred = self.forward_pass(input[0], True)
			# print("\nb")
			# print(input[1].shape)
			# print(pred.shape)
			loss = self.compiled_loss(input[1], pred)
			# print("\nc")

		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		self.compiled_metrics.update_state(input[1], pred)

		return {m.name: m.result() for m in self.metrics}


	def test_step(self, input):

		pred = self.forward_pass(input[0], False)
		loss = self.compiled_loss(input[1], pred)

		self.compiled_metrics.update_state(input[1], pred)

		return {m.name: m.result() for m in self.metrics}


	def call(self, input, training=False):

		return self.forward_pass(input, training)
