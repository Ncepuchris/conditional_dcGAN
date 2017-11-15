#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:34:53 2017

@author: no1

y表示数据标签，z表示均匀分布的噪声，images表示mnist图片
"""
import tensorflow as tf
import model
import dataset
import time
from datetime import datetime
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
batch_size = model.batch_size
y_dim = model.y_dim
noise_length = 100
noise_num = 64
train_dir = 'summary'
checkpoint_dir = 'ckpt'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
IMAGE_HEIGHT = model.IMAGE_HEIGHT
IMAGE_WIDTH = model.IMAGE_WIDTH

def get_demo_label(num = 100):
  onehot_labels = np.zeros(shape = (num, 10))
  for i in range(num):
    onehot_labels[i, i//10] = 1
  return onehot_labels


def placeholder_inputs():
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,IMAGE_HEIGHT, IMAGE_WIDTH, 1))
  y_placeholder = tf.placeholder(tf.float32, shape=(batch_size, y_dim))
  z_placeholder = tf.placeholder(tf.float32, shape=(None, noise_length))
  return images_placeholder, y_placeholder, z_placeholder

def fill_feed_dict(data_set, images1_pl, labels_pl, noise_pl):

  images_feed, labels_feed = data_set.next_batch(batch_size,False)
  noise = np.random.uniform(-1, 1, size =(batch_size, noise_length))
  feed_dict = {
      images1_pl: images_feed,
      labels_pl: labels_feed,
      noise_pl: noise
  }

  return feed_dict
def save_images(samples, step):

  fig,axes=plt.subplots(nrows=10,ncols=10,figsize=(14,14))
  for img,ax in zip(samples,axes.flatten()):
    ax.imshow(img.reshape((28,28)),cmap='Greys_r')
    ax.axis('off')
  fig.tight_layout()
  plt.subplots_adjust(wspace =0, hspace =0)

  plt.savefig('demo_result/demo_{:05d}.png'.format(step))
def run_train():
  """Train CAPTCHA for a number of steps."""
  train_data = dataset.read_data_sets()

  with tf.Graph().as_default():
    images_placeholder, y_placeholder,z_placeholder = placeholder_inputs()

    d_logits_real, d_logits_fake = model.inference(images_placeholder,
                                                   z_placeholder, y_placeholder)
    demo_noise = np.random.uniform(-1, 1, size = (100,100))
    demo_label = get_demo_label()
    demo_img = model.generator(z_placeholder, y_placeholder, reuse = True)

    g_loss, d_loss = model.loss(d_logits_real, d_logits_fake)
    tf.summary.scalar('g_loss', g_loss)
    tf.summary.scalar('d_loss', d_loss)
    summary = tf.summary.merge_all()
    train_op = model.get_optimizer(g_loss, d_loss)
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    sess.run(init_op)

    try:
      max_step = 100 * 70000 // batch_size
      for step in range(1,max_step):
        start_time = time.time()
        feed_dict = fill_feed_dict(train_data, images_placeholder,
                                   y_placeholder, z_placeholder)

        _, gloss_value, dloss_value = sess.run([train_op, g_loss, d_loss], feed_dict = feed_dict)

        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

        duration = time.time() - start_time
        if step % 10 == 0:
          print('>> Step %d run_train: g_loss = %.2f d_loss = %.2f(%.3f sec)'
                % (step, gloss_value, dloss_value, duration))
          #-------------------------------

        if step % 100 == 0:

          demo_result = sess.run(demo_img, feed_dict = {z_placeholder:demo_noise,y_placeholder:demo_label })
          save_images(demo_result, step)
          print('>> %s Saving in %s' % (datetime.now(), checkpoint_dir))
          saver.save(sess, checkpoint_file, global_step=step)

    except KeyboardInterrupt:
      print('INTERRUPTED')

    finally:
      saver.save(sess, checkpoint_file, global_step=step)
      print('Model saved in file :%s'%checkpoint_dir)

    sess.close()





if __name__ == '__main__':
  run_train()







