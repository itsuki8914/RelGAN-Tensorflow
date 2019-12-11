import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt

REGULARIZER_COF = 1e-6
"""
#for bs >1
def _norm(x,name="BN",isTraining=True):
    bs, h, w, c = x.get_shape().as_list()

    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None,
                                        epsilon=1e-5, scale=True,
                                        is_training=isTraining, scope="BN"+name)
"""
#for bs =1
def _norm(x,name="BN",isTraining=True):
    bs, h, w, c = x.get_shape().as_list()
    s = tf.get_variable(name+"s", c,
                        initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    o = tf.get_variable(name+"o", c,
                        initializer=tf.constant_initializer(0.0))
    mean, var = tf.nn.moments(x, axes=[1,2], keep_dims=True)
    eps = 10e-10
    normalized = (x - mean) / (tf.sqrt(var) + eps)
    return s * normalized + o


def _fc_variable( weight_shape,name="fc"):
    with tf.variable_scope(name):
        # check weight_shape
        input_channels  = int(weight_shape[0])
        output_channels = int(weight_shape[1])
        weight_shape    = (input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)

        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer =regularizer)
        bias   = tf.get_variable("b", [weight_shape[1]],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv_variable( weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [output_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _deconv_variable( weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        output_channels = int(weight_shape[2])
        input_channels  = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape    ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [input_channels], initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv2d( x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def _deconv2d( x, W, output_shape, stride=1):
    # x           : [nBatch, height, width, in_channels]
    # output_shape: [nBatch, height, width, out_channels]
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")

def _conv_layer(x, input_layer, output_layer, stride, filter_size=3, name="conv", isTraining=True):
    conv_w, conv_b = _conv_variable([filter_size,filter_size,input_layer,output_layer],name="conv"+name)
    h = _conv2d(x,conv_w,stride=stride) + conv_b
    h = _norm(h,name)
    h = tf.nn.leaky_relu(h)
    return h

def _deconv_layer(x,input_layer, output_layer, stride=2, filter_size=3, name="deconv", isTraining=True):
    bs, h, w, c = x.get_shape().as_list()
    deconv_w, deconv_b = _deconv_variable([filter_size,filter_size,input_layer,output_layer],name="deconv"+name )
    h = _deconv2d(x,deconv_w, output_shape=[bs,h*2,w*2,output_layer], stride=stride) + deconv_b
    h = _norm(h,name)
    h = tf.nn.leaky_relu(h)
    return h

def buildGenerator(x,label_A2B,num_domains,reuse=False,isTraining=True,ksize=3,resBlock=12,name="generator"):

    bs, h, w, c = x.get_shape().as_list()
    #l = tf.one_hot(label_A2B,num_domains,name="label_onehot")
    #l = tf.reshape(l,[int(bs),1,1,num_domains])
    l = tf.reshape(label_A2B,[int(bs),1,1,num_domains])
    k = tf.ones([int(bs),int(h),int(w),int(num_domains)])
    k = k * l
    x = tf.concat([x,k],axis=3)

    with tf.variable_scope(name, reuse=reuse) as scope:
        if reuse: scope.reuse_variables()

        conv_w, conv_b = _conv_variable([7,7,3+num_domains,64],name="conv4-2_g")
        h = _conv2d(x,conv_w,stride=1) + conv_b
        h = tf.nn.leaky_relu(h)

        h = _conv_layer(h,64,128,2,ksize,"4-1_g")
        h = _conv_layer(h,128,256,2,ksize,"4-0_g")

        for i in range(resBlock):
            conv_w, conv_b = _conv_variable([ksize,ksize,256,256],name="res%s-1" % i)
            nn = _conv2d(h,conv_w,stride=1) + conv_b
            nn = _norm(h,name="Norm%s-1_g" %i)
            nn = tf.nn.leaky_relu(nn)
            conv_w, conv_b = _conv_variable([ksize,ksize,256,256],name="res%s-2" % i)
            nn = _conv2d(nn,conv_w,stride=1) + conv_b
            nn = _norm(h,name="Norm%s-2_g" %i)

            nn = tf.math.add(h,nn, name="resadd%s" % i)
            h = nn

        h = _deconv_layer(h, 256, 128, 2, ksize, "2-2_g")

        h = _deconv_layer(h, 128, 64, 2, ksize, "2-1_g")

        #h = tf.math.add(tmp,h, name="add1")
        conv_w, conv_b = _conv_variable([7,7,64,3],name="convo_g" )
        h = _conv2d(h,conv_w,stride=1) + conv_b
        y = tf.nn.tanh(h)

    return y

def _conv_layer_dis(x,input_layer, output_layer, stride, filter_size=3, name="conv", isTraining=True):
    conv_w, conv_b = _conv_variable([filter_size,filter_size,input_layer,output_layer],name="conv"+name)
    h = _conv2d(x,conv_w,stride=stride) + conv_b
    #h = _norm(h,name)
    h = tf.nn.leaky_relu(h)
    return h


def buildDiscriminator(y1, y2, num_domains, reuse=False, ksize=4,name="discriminator"):
    with tf.variable_scope(name) as scope:
        if reuse: scope.reuse_variables()
        fn_l = 32
        h = y

        # conv1
        h = _conv_layer_dis(h, 3, fn_l, 2, ksize, "1-1di")
        # conv2
        h = _conv_layer_dis(h, fn_l, fn_l*2, 2, ksize, "2-1di")
        # conv3
        h = _conv_layer_dis(h, fn_l*2, fn_l*4, 2, ksize, "3-1di")
        # conv4
        h = _conv_layer_dis(h, fn_l*4, fn_l*8, 2, ksize, "4-1di")
        # conv5
        h = _conv_layer_dis(h, fn_l*8, fn_l*16, 2, ksize, "5-1di")

        # conv6
        conv_w, conv_b = _conv_variable([ksize,ksize,fn_l*16,1+num_domains],name="conv5di")
        h = _conv2d(h,conv_w, stride=1) + conv_b
        h = tf.reshape(tf.reduce_mean(h,axis=[1,2]),[-1,1+num_domains])

        src = h[:,0]
        cls = h[:,1:]
        cls = tf.nn.softmax(cls)

    return src, cls
