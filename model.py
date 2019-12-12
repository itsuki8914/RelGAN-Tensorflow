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
    eps = 1e-8
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

def _resBlk(x, input_layer, output_layer, filter_size=3, num=0):
    conv_w, conv_b = _conv_variable([3, 3, input_layer, output_layer], name="res%s-1" % num)
    nn = _conv2d(x, conv_w, stride=1) + conv_b
    nn = _norm(nn, name="Norm%s-1_g" %num)
    nn = tf.nn.leaky_relu(nn)
    conv_w, conv_b = _conv_variable([3, 3, output_layer, output_layer],name="res%s-2" % num)
    nn = _conv2d(nn,conv_w,stride=1) + conv_b
    nn = _norm(nn,name="Norm%s-2_g" %num)

    #nn = tf.math.add(h,nn, name="resadd-%s" % i)
    nn += x
    return nn

def _deconv_layer(x,input_layer, output_layer, stride=2, filter_size=4, name="deconv", isTraining=True):
    bs, h, w, c = x.get_shape().as_list()
    deconv_w, deconv_b = _deconv_variable([filter_size,filter_size,input_layer,output_layer],name="deconv"+name )
    h = _deconv2d(x,deconv_w, output_shape=[bs,h*stride,w*stride,output_layer], stride=stride) + deconv_b
    h = _norm(h,name)
    h = tf.nn.leaky_relu(h)
    return h

def buildGenerator(x,label_A2B,num_domains,reuse=False,isTraining=True,ksize=3,resBlock=6,name="generator"):

    bs, h, w, c = x.get_shape().as_list()
    #print(x.get_shape().as_list())

    #bs, h, w, c = tf.shape(x)
    #l = tf.one_hot(label_A2B,num_domains,name="label_onehot")
    #l = tf.reshape(l,[int(bs),1,1,num_domains])
    l = tf.reshape(label_A2B,[int(bs),1,1,num_domains])
    k = tf.ones([int(bs),int(h),int(w),int(num_domains)])
    k = k * l
    x = tf.concat([x,k],axis=3)

    with tf.variable_scope(name, reuse=reuse) as scope:
        if reuse: scope.reuse_variables()

        h = _conv_layer(x, 3+num_domains, 64, 1, 7, "ds1")
        h = _conv_layer(h, 64, 128, 2, 3, "ds2")
        h = _conv_layer(h, 128, 256, 2, 3, "ds3")

        for i in range(resBlock):
            h = _resBlk(h, 256, 256, 3, i)


        h = _deconv_layer(h, 256, 128, 2, 4, "us3")

        h = _deconv_layer(h, 128, 64, 2, 4, "us2")

        #h = tf.math.add(tmp,h, name="add1")
        conv_w, conv_b = _conv_variable([7,7,64,3],name="us1")
        h = _conv2d(h,conv_w,stride=1) + conv_b
        y = tf.nn.tanh(h)

    return y

def _conv_layer_dis(x,input_layer, output_layer, stride, filter_size=3, name="conv", isTraining=True):
    conv_w, conv_b = _conv_variable([filter_size,filter_size,input_layer,output_layer],name="conv"+name)
    h = _conv2d(x,conv_w,stride=stride) + conv_b
    #h = _norm(h,name)
    h = tf.nn.leaky_relu(h)
    return h


def buildDiscriminator(y1, y2, vec, num_domains, reuse=[1,1], method="adv"):
    fn_l = 32
    def feature_layer(y):
        with tf.variable_scope("discriminator") as scope:
            if reuse[0]: scope.reuse_variables()

            # conv1
            h = _conv_layer_dis(y, 3, fn_l, 2, 4, "fl1")
            # conv2
            h = _conv_layer_dis(h, fn_l, fn_l*2, 2, 4, "fl2")
            # conv3
            h = _conv_layer_dis(h, fn_l*2, fn_l*4, 2, 4, "fl3")
            # conv4
            h = _conv_layer_dis(h, fn_l*4, fn_l*8, 2, 4, "fl4")
            # conv5
            h = _conv_layer_dis(h, fn_l*8, fn_l*16, 2, 4, "fl5")
            # conv6
            #h = _conv_layer_dis(h, fn_l*16, fn_l*32, 2, 4, "fl6")
        return h

    if method == "adv":
        h = feature_layer(y1)
        with tf.variable_scope("adv") as scope:
            if reuse[1]: scope.reuse_variables()
            conv_w, conv_b = _conv_variable([1,1,fn_l*16,1],name="adv")
            adv = _conv2d(h,conv_w, stride=1) + conv_b
            return adv

    if method == "int":
        h = feature_layer(y1)
        with tf.variable_scope("interpolation") as scope:
            if reuse[1]: scope.reuse_variables()
            conv_w, conv_b = _conv_variable([1,1,fn_l*16,fn_l],name="int")
            interp = _conv2d(h,conv_w, stride=1) + conv_b
            interp = tf.reduce_mean(interp, axis=3)
            return interp

    if method == "mat":
        assert y2 != None
        h1 = feature_layer(y1)
        h2 = feature_layer(y2)
        with tf.variable_scope("match") as scope:
            if reuse[1]: scope.reuse_variables()
            bs, h, w, c = h1.get_shape().as_list()
            atr = tf.tile(vec, [h,w])
            atr = tf.reshape(atr,[-1,h, w, num_domains])
            h = tf.concat([h1, h2, atr], axis=3)
            conv_w, conv_b = _conv_variable([1,1,fn_l*32+num_domains,fn_l*16],name="mat1")
            mat = _conv2d(h,conv_w, stride=1) + conv_b
            h = tf.nn.leaky_relu(h)
            conv_w, conv_b = _conv_variable([1,1,fn_l*16,1],name="mat2")
            mat = _conv2d(mat,conv_w, stride=1) + conv_b
            return mat
