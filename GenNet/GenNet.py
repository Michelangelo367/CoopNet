from __future__ import division

import os
import math
import numpy as np
import tensorflow as tf

from ops import leaky_relu
from datasets import *


class GenNet(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.delta = config.delta
        self.sigma = config.sigma
        self.sample_steps = config.sample_steps
        self.z_dim = config.z_dim

        self.num_epochs = config.num_epochs
        self.data_path = os.path.join(config.data_path, config.category)
        self.log_step = config.log_step
        self.output_dir = os.path.join(config.output_dir, config.category)

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.sample_dir = os.path.join(self.output_dir, 'sample')
        self.model_dir = os.path.join(self.output_dir, 'checkpoints')

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32)
        self.z = tf.placeholder(shape=[None, self.z_dim], dtype=tf.float32)
        self.build_model()


    def generator(self, inputs, reuse=False, is_training=True):
        ####################################################
        # Define the structure of generator, you may use the
        # generator structure of DCGAN. ops.py defines some
        # layers that you may use.
        ####################################################
        deconv = tf.contrib.layers.conv2d_transpose
        linear = tf.contrib.layers.fully_connected
        batch_norm = lambda input_, name: tf.contrib.layers.batch_norm(input_, decay=0.9,
                updates_collections=None, scale=True, is_training=is_training, scope=name)
        n = tf.shape(inputs)[0]
        with tf.variable_scope('generator', reuse=reuse):
            outputs = batch_norm(linear(inputs, 4*16**2, activation_fn=leaky_relu), 'bn1')
            outputs = tf.reshape(outputs, (n, 8, 8, 16))
            outputs = batch_norm(deconv(outputs, 64, 2, stride=2, activation_fn=leaky_relu), 'bn2')
            outputs = batch_norm(deconv(outputs, 128, 2, stride=2, activation_fn=leaky_relu), 'bn3')
            #outputs = batch_norm(deconv(outputs, 256, 2, stride=2, activation_fn=leaky_relu), 'bn4')
            outputs = deconv(outputs, 3, 2, stride=2, activation_fn=tf.tanh)
            return outputs


    def langevin_dynamics(self, z):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated z.
        ####################################################
        _, z = tf.while_loop(lambda i, z: tf.less(i, self.sample_steps), self._langevin, [0, z])
        return z

    # helper function for langevin_dynamics
    def _langevin(self, i, z):
        u_t = self.sigma * tf.random_normal(tf.shape(z))
        out_t = self.generator(z, reuse=True, is_training=False)
        grad_t, = tf.gradients(-tf.reduce_mean(tf.square(self.obs - out_t)), z)
        z = z + u_t + self.delta**2 * (grad_t / (2 * self.sigma**2) - z) / 2
        return i + 1, z


    def build_model(self):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        self.synth_images = self.generator(self.z) # initialize model
        self.sample_z = self.langevin_dynamics(self.z)
        self.loss = tf.reduce_sum(tf.square(self.obs - self.synth_images)) / (2 * self.sigma**2)

        self.global_step = tf.get_variable('global_step', (),
                initializer=tf.zeros_initializer(), trainable=False)
        generator_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
        self.optimizer = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1).minimize(
                self.loss, global_step=self.global_step, var_list=generator_vars)


    def train(self):
        # Prepare training data
        train_data = DataSet(self.data_path, image_size=self.image_size)
        train_data = train_data.to_range(-1, 1)

        num_batches = int(math.ceil(len(train_data) / self.batch_size))
        summary_op = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.sess.graph.finalize()

        print('Start training ...')

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # reconstructed images and synthesized images in
        # self.sample_dir, loss in self.log_dir (using writer).
        ####################################################
        f = open('%s/cat.log' % self.log_dir, 'w+')
        inter_z = np.mgrid[-2:2.1:0.5, -2:2.1:0.5].reshape(2, -1).T
        synth_z = np.random.normal(size=(9**2, self.z_dim))
        init_z = np.random.normal(size=(self.batch_size, self.z_dim))
        save_images(train_data, "%s/original.png" % self.sample_dir)
        step = 0
        while step < self.num_epochs:
            # no actual batches
            init_z = self.sess.run(self.sample_z, feed_dict={self.obs: train_data, self.z: init_z})
            feed_dict = {self.obs : train_data, self.z : init_z}
            step, loss, recon_images, _ = self.sess.run([self.global_step, self.loss,
                    self.synth_images, self.optimizer], feed_dict=feed_dict)


            f.write("%f\n" % loss)
            if step % self.log_step == 0 or step == 1:
                print("Step %d: %2.6f" % (step, loss))
                saver.save(self.sess, "%s/model_%d.ckpt" % (self.model_dir, step))
                synth_images = self.sess.run(self.synth_images, feed_dict={self.z : synth_z})
                inter_images = self.sess.run(self.synth_images, feed_dict={self.z : inter_z})
                save_images(recon_images, "%s/reconstructed_%d.png" % (self.sample_dir, step), 1)
                save_images(synth_images, "%s/synthetic_%d.png" % (self.sample_dir, step), 1)
                save_images(inter_images, "%s/interpolated_%d.png" % (self.sample_dir, step), 1)