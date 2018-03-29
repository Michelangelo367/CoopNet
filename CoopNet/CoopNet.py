import os
import math
import numpy as np
import tensorflow as tf

from datasets import *
from tensorflow.contrib.layers import fully_connected, conv2d, conv2d_transpose, batch_norm, flatten

def leaky_relu(input_, leak=0.2):
	assert leak < 1
	return tf.maximum(leak*input_, input_)


class CoopNet(object):
	def __init__(self, sess, config, restore=False):
		self.sess = sess
		self.image_size = config.image_size
		self.c_dim = config.color_dim
		self.latent_dim = config.latent_dim
		
		# sampling values
		self.sampling_iters = config.sample_steps
		self.delta = config.delta
		self.sigma = config.sigma

		# training values
		self.g_lr = config.gen_lr
		self.d_lr = config.des_lr
		self.beta1 = config.beta1

		# load data
		train_data = DataSet(os.path.join(config.data_path, config.category))
		self.train_data = train_data.to_range(-1, 1)
		self.sample_size = self.train_data.shape[0]

		# create placeholders
		self.real_data = tf.placeholder(shape=[None, self.image_size, self.image_size, self.c_dim], dtype=tf.float32, name='real')
		self.dream_data = tf.placeholder(shape=[None, self.image_size, self.image_size, self.c_dim], dtype=tf.float32, name='dream')
		self.latent_vector = tf.placeholder(shape=[None, self.latent_dim], dtype=tf.float32, name='latent')
		
		# define model, sampling operations, and training operations
		self._build_model()

		# directory paths
		self.log_dir = os.path.join(config.category, 'log')
		self.sample_dir = os.path.join(config.category, 'samples')
		self.model_dir = os.path.join(config.category, 'checkpoints')
		self.log_epoch = config.log_epoch

		try:
			if not restore:
				raise Exception

			print("Attempting to restore weights")
			# get most recent 
		except Exception:
			# create directories
			if not os.path.exists(self.log_dir):
				os.makedirs(self.log_dir)
			if not os.path.exists(self.sample_dir):
				os.makedirs(self.sample_dir)
			if not os.path.exists(self.model_dir):
				os.makedirs(self.model_dir)

			# initialize variables
			self.epoch = 0
			self.saver = tf.train.Saver(max_to_keep=50)
			self.sess.run(tf.global_variables_initializer())
			self.sess.run(tf.local_variables_initializer())


	def generator(self, input_, reuse=True, is_training=True):
		# define descriptor network
		bn = lambda input_, name: tf.contrib.layers.batch_norm(input_, decay=0.9,
				updates_collections=None, scale=True, is_training=is_training, scope=name)
		n = tf.shape(input_)[0]
		with tf.variable_scope('generator', reuse=reuse):
			outputs = bn(fully_connected(input_, 4*16**2, activation_fn=leaky_relu), 'bn1')
			outputs = tf.reshape(outputs, (n, 8, 8, 16))
			outputs = bn(conv2d_transpose(outputs, 64, 2, stride=2, activation_fn=leaky_relu), 'bn2')
			outputs = bn(conv2d_transpose(outputs, 128, 2, stride=2, activation_fn=leaky_relu), 'bn3')
			#outputs = bn(conv2d_transpose(outputs, 256, 2, stride=2, activation_fn=leaky_relu), 'bn4')
			outputs = conv2d_transpose(outputs, self.c_dim, 2, stride=2, activation_fn=tf.tanh)
			return outputs


	def descriptor(self, input_, reuse=True, is_training=True):
		bn = lambda input_: tf.contrib.layers.batch_norm(input_, decay=0.9, updates_collections=None,
				scale=True, is_training=is_training)
		with tf.variable_scope('descriptor', reuse=reuse) as scope:
			outputs = leaky_relu(conv2d(input_, 64, 4, stride=2, activation_fn=batch_norm))
			outputs = leaky_relu(conv2d(outputs, 128, 2, stride=1, activation_fn=batch_norm))
			#outputs = leaky_relu(tf.contrib.layers.fully_connected(outputs, 100, activation_fn=bn))
			outputs = flatten(outputs)
			outputs = fully_connected(outputs, self.latent_dim, activation_fn=None)
			return outputs


	def _langevin_sample_vector(self, vector):
		cond = lambda i, vec: tf.less(i, self.sampling_iters)
		def body(i, vec):
			u_t = self.sigma * tf.random_normal(tf.shape(vec))
			out_t = self.generator(vec, is_training=False)
			grad_t, = tf.gradients(-tf.reduce_mean(tf.square(self.real_data - out_t)), vec)
			vec_t = vec + u_t + self.delta**2 * (grad_t / (2 * self.sigma**2) - vec) / 2
			return i+1, vec

		_, vec_t = tf.while_loop(cond, body, [0, vector])
		return vec_t


	def _langevin_sample_image(self, image):
		cond = lambda i, img: tf.less(i, self.sampling_iters)
		def body(i, img):
			u_t = self.sigma * tf.random_normal(tf.shape(img))
			grad_t, = tf.gradients(self.descriptor(img), img)
			img = img + u_t - self.delta**2 * grad_t / 2
			return i+1, img

		_, img_t = tf.while_loop(cond, body, [0, image])
		return img_t


	def _build_model(self):
		# generator and descriptor network initialization
		self.reconstructed_images = self.generator(self.latent_vector, reuse=False)
		self.labels = self.descriptor(self.dream_data, reuse=False)

		# langevin sampling operations
		self.sample_image_op = self._langevin_sample_image(self.dream_data)
		self.sample_vector_op = self._langevin_sample_vector(self.latent_vector)

		# define generator and descriptor loss
		self.generator_loss = tf.reduce_mean(tf.square(self.real_data - self.reconstructed_images) / (2 * self.sigma**2))
		self.descriptor_loss = tf.reduce_mean(tf.square(self.labels) - tf.square(self.descriptor(self.real_data)))

		# define training operations
		t_vars = tf.trainable_variables()
		g_vars = [var for var in t_vars if 'generator' in var.name]
		d_vars = [var for var in t_vars if 'descriptor' in var.name]
		self.generator_train_op = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1) \
				.minimize(self.generator_loss, var_list=g_vars)
		self.descriptor_train_op = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1) \
				.minimize(self.descriptor_loss, var_list=d_vars)


	def train(self, epochs):
		assert self.epoch >= 0
		print("Starting training...")

		inter_vec = np.mgrid[-2:2.1:0.5, -2:2.1:0.5].reshape(2, -1).T
		while self.epoch < epochs:
			random_vector = np.random.normal(size=(self.sample_size, self.latent_dim))
			
			# use langevin sampling to generate data
			init_images = self.sess.run(self.reconstructed_images, feed_dict={self.latent_vector: random_vector})
			dream_images = self.sess.run(self.sample_image_op, feed_dict={self.dream_data: init_images})
			init_labels = self.sess.run(self.labels, feed_dict={self.dream_data : dream_images})
			dream_labels = self.sess.run(self.sample_vector_op, feed_dict={self.latent_vector: init_labels, self.real_data: self.train_data})

			# use dream data to optimize networks
			d_loss, _ = self.sess.run([self.descriptor_loss, self.descriptor_train_op], feed_dict={self.dream_data: dream_images, self.real_data: self.train_data})
			g_loss, recon_images, _ = self.sess.run([self.generator_loss, self.reconstructed_images, self.generator_train_op], feed_dict={self.latent_vector: dream_labels, self.real_data: self.train_data})
			self.epoch += 1

			print(" Step: %d, D: %f, G: %f" % (self.epoch, d_loss, g_loss))
			if self.epoch % self.log_epoch == 0 or self.epoch == 1:
				self.saver.save(self.sess, "%s/model_%d.ckpt" % (self.model_dir, self.epoch))
				inter_images = self.sess.run(self.reconstructed_images, feed_dict={self.latent_vector: inter_vec})
				save_images(dream_images, "%s/dream_%d.png" % (self.sample_dir, self.epoch), 1)
				save_images(inter_images, "%s/inter_%d.png" % (self.sample_dir, self.epoch), 1)
