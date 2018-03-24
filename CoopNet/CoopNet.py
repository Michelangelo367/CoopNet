import os
import math
import numpy as np
import tensorflow as tf

class CoopNet(object):
	def __init__(self, sess, config):
		self.sess = sess
		self.build_model()


	def generator(self, input, reuse=True):
		with tf.variable_scope('generator', reuse=reuse):
			pass # define network


	def descriptor(self, input, reuse=True):
		with tf.variable_scope('descriptor', reuse=reuse):
			pass # define descriptor


	def _langevin_sample_vector(self, image):
		cond = lambda i, img: tf.less(i, self.sample_vector_steps)
		def body(i, img):
			# define langevin dynamics

		_, img_t = tf.while_loop(cond, body, [0, image])
		return img_t


	def _langevin_sample_image(self, vector):
		cond = lambda i, vec: tf.less(i, self.sample_image_steps)
		def body(i, img):
			# define langevin dynamics

		_, vec_t = tf.while_loop(cond, body, [0, vector])
		return vec_t


	def build_model(self):
		pass


	def train(self):
		pass


	def save(self):
		pass


	def restore(self):
		pass