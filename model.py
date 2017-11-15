#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:05:02 2017

@author: no1
"""
import tensorflow as tf

batch_size = 100
rate = 0.01
y_dim = 10
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
gen_weights = {'deconv1':1024, 'deconv2':7*7*2*64,
               'deconv3':128, 'deconv4': 1}
dis_weights = {'conv1':10, 'conv2':64, 'f1':1024, 'f2':1}
def concat_matrix(x, y):

  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1],
                                  x_shapes[2], y_shapes[3]])], 3)

def generator(placeholder_noise, placeholder_y, reuse):
  with tf.variable_scope('generator') as scope:
    if reuse:
      scope.reuse_variables()

    y_matrix = tf.reshape(placeholder_y, shape = (batch_size, 1, 1, y_dim))

    z = tf.concat([placeholder_noise, placeholder_y], 1)

    deconv1 = tf.layers.dense(z, gen_weights['deconv1'])#(batch,1024)
    deconv1 = tf.layers.batch_normalization(deconv1, training = not reuse)
    deconv1 = tf.maximum(rate * deconv1, deconv1)
    deconv1 = tf.concat([deconv1, placeholder_y], 1) #(batch,1034)

    deconv2 = tf.layers.dense(deconv1, gen_weights['deconv2'])
    deconv2 = tf.layers.batch_normalization(deconv2, training = not reuse)
    deconv2 = tf.maximum(rate*deconv2, deconv2)
    deconv2 = tf.reshape(deconv2, [batch_size, 7, 7, 64*2])  #(batch,7,7,128)
    deconv2 = concat_matrix(deconv2, y_matrix)  #(batch,7,7,138)

    deconv3 = tf.layers.conv2d_transpose(deconv2, gen_weights['deconv3'],
                        kernel_size = 5, strides =2, padding ='SAME')#(batch,14,14,128)
    deconv3 = tf.layers.batch_normalization(deconv3, training = not reuse)
    deconv3 = tf.maximum(rate * deconv3, deconv3)
    deconv3 = concat_matrix(deconv3, y_matrix)#(batch,14,14,138)

    deconv4 = tf.layers.conv2d_transpose(deconv3, gen_weights['deconv4'],
                         kernel_size = 5, strides = 2, padding = 'SAME')#(batch,28,28,1)
    return tf.nn.sigmoid(deconv4)





def descriminator(placeholder_images, placeholder_y, reuse):
  with tf.variable_scope('descriminator') as scope:
    if reuse:
      scope.reuse_variables()

    y_matrix = tf.reshape(placeholder_y, shape = (batch_size, 1,1,y_dim))

    images = concat_matrix(placeholder_images, y_matrix) #(batch,28,28,11)
    conv1 = tf.layers.conv2d(images, dis_weights['conv1'], kernel_size = 5,
                             strides = 2, padding = 'SAME')#(batch,14,14,10)
    conv1 = tf.maximum(rate * conv1, conv1)
    conv1 = concat_matrix(conv1, y_matrix) #(batch, 14,14,20)


    conv2 = tf.layers.conv2d(conv1, dis_weights['conv2'], kernel_size = 5,
                             strides = 2, padding = 'SAME') #(batch,7,7,64)
    conv2 = tf.layers.batch_normalization(conv2, training = True)
    conv2 = tf.maximum(rate * conv2, conv2)
    conv2 = tf.reshape(conv2, shape = [batch_size,-1]) #(batch,7*7*64)
    conv2 = tf.concat([conv2, placeholder_y], 1) #(batch,7*7*64+10)

    f1 = tf.layers.dense(conv2, dis_weights['f1']) #(batch,1034)
    f1 = tf.layers.batch_normalization(f1, training = True)
    f1 = tf.maximum(rate * f1, f1)
    f1 = tf.concat([f1, placeholder_y], axis = 1) #(batch,1034)

    f2 = tf.layers.dense(f1,dis_weights['f2'])
    return f2

def inference(real_images, placeholder_noise, placeholder_y):
  fake_images = generator(placeholder_noise, placeholder_y, False)
  d_logits_real = descriminator(real_images, placeholder_y, False)
  d_logits_fake = descriminator(fake_images, placeholder_y, True)

  return d_logits_real, d_logits_fake

def loss(d_logits_real, d_logits_fake, smooth =0.05):

  g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                        labels=tf.ones_like(d_logits_fake)*(1-smooth)))

  d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                        labels=tf.ones_like(d_logits_real)*(1-smooth)))
  d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                        labels=tf.zeros_like(d_logits_fake)))

  d_loss = tf.add(d_loss_real, d_loss_fake)

  return g_loss, d_loss



def get_optimizer(g_loss, d_loss,beta=0.4,learning_rate=0.001):

  g_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
  d_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='descriminator')

  g_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta,name='G_opt').minimize(g_loss,var_list=g_vars)
  d_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta,name='D_opt').minimize(d_loss,var_list=d_vars)
  with tf.control_dependencies([g_opt, d_opt]):
    return tf.no_op(name='optimizers')



























