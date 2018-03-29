# -*- coding: utf-8 -*-
import os
from glob import glob
import time
import tensorflow as tf
import numpy as np

from datasets import *

class DesNet(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.batch_size = flags.batch_size

        self.data_path = os.path.join('./Image', flags.dataset_name)
        self.output_dir = flags.output_dir

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

        self.build_model(flags)

    def descriptor(self, inputs, is_training=True, reuse=False):
        ####################################################
        # Define network structure for descriptor.
        # Recommended structure:
        # conv1: channel 64 kernel 4*4 stride 2
        # conv2: channel 128 kernel 2*2 stride 1
        # fc: channel output 1
        # conv1 - bn - relu - conv2 - bn - relu -fc
        ####################################################
        bn = lambda input_: tf.contrib.layers.batch_norm(input_, decay=0.9, updates_collections=None,
                scale=True, is_training=is_training)
        with tf.variable_scope('descriptor', reuse=reuse):
            outputs = tf.nn.relu(tf.contrib.layers.conv2d(inputs, 64, 4, stride=2, activation_fn=bn))
            outputs = tf.nn.relu(tf.contrib.layers.conv2d(outputs, 128, 2, stride=1, activation_fn=bn))
            #outputs = bn(tf.contrib.layers.fully_connected(outputs, 1000))
            outputs = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=tf.sigmoid)
            return outputs

    def Langevin_sampling(self, samples, flags):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated samples.
        ####################################################
        cond = lambda i, s : tf.less(i, flags.T)
        body = lambda i, s : self._langevin(i, s, flags)
        _, samples = tf.while_loop(cond, body, [0, samples])
        return samples

    # langevin sampling helper function
    def _langevin(self, i, samples, flags):
        u_t = flags.delta * tf.random_normal(tf.shape(samples))
        grad_t, = tf.gradients(self._energy(samples, flags), samples)
        samples_t = samples - flags.delta**2 * grad_t / 2 + u_t
        return i+1, samples_t

    def build_model(self, flags):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        color_channels = 3
        self.Y = tf.placeholder(shape=[None, flags.image_size, flags.image_size, color_channels], dtype=tf.float32)
        self.Y_ = tf.placeholder(shape=[None, flags.image_size, flags.image_size, color_channels], dtype=tf.float32)

        self.z = self.descriptor(self.Y) # init
        self.synth_Y = self.Langevin_sampling(self.Y_, flags)
        self.loss = self._energy(self.Y, flags) - self._energy(self.Y_, flags)

        self.global_step = tf.get_variable('global_step', (),
                initializer=tf.zeros_initializer(), trainable=False)
        descriptor_vars = [var for var in tf.trainable_variables() if 'descriptor' in var.name]
        self.optimizer = tf.train.GradientDescentOptimizer(flags.learning_rate).minimize(
                self.loss, global_step=self.global_step, var_list=descriptor_vars)
    
    # energy helper function
    def _energy(self, samples, flags):
        image_energy = tf.reduce_sum(samples, [1, 2, 3]) / (2 * flags.ref_sig**2)
        descriptor_energy = self.descriptor(samples, reuse=True)
        return tf.reduce_mean(image_energy - descriptor_energy)

    def train(self, flags):
        # Prepare training data, scale is [0, 255]
        train_data = DataSet(self.data_path, image_size=flags.image_size)
        train_data = train_data.to_range(0, 255)

        saver = tf.train.Saver(max_to_keep=50)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.sess.graph.finalize()

        print(" Start training ...")

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # synthesized images in self.sample_dir,
        # loss in self.log_dir (using writer).
        ####################################################
        f = open('%s/cat.log' % self.log_dir, 'w+')
        mean_data = np.repeat(train_data.mean(1).mean(1), 64*64, axis=0).reshape(train_data.shape)
        save_images(train_data, "%s/original.png" % self.sample_dir)
        for epoch in range(flags.epoch):
            synth_images = self.sess.run(self.synth_Y, feed_dict={self.Y_ : mean_data})
            feed_dict = {self.Y : train_data, self.Y_ : synth_images}
            step, loss, _ = self.sess.run([self.global_step, self.loss, self.optimizer], feed_dict=feed_dict)

            f.write("%f\n" % loss)
            if step % 10 == 0 or step == 1:
                print("Step %d: %f" % (step, loss))
                saver.save(self.sess, "%s/model_%d.ckpt" % (self.model_dir, step))
                save_images(synth_images, "%s/synth_%d.png" % (self.sample_dir, step), 1)
