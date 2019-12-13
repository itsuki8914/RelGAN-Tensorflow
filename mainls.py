import os, sys, shutil, glob
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model import *
from btgen import *

SAVE_DIR = "model"
SVIM_DIR = "samples"
TRAIN_DIR = "train"
VAL_DIR = "test"

def tileImage(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d,w*d,3),dtype=np.float32)
    for idx,img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
    return r

def foloderLength(folder):
    dir = folder
    paths = os.listdir(dir)
    return len(paths)

def printParam(scope):
    total_parameters = 0
    for variable in tf.trainable_variables(scope=scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("{} has {} parameters".format(scope, total_parameters))

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(SVIM_DIR):
        os.makedirs(SVIM_DIR)
    img_size = 128
    bs = 8
    lr = tf.placeholder(tf.float32)

    trans_lr = 1e-4
    max_step = 100000
    gp_lmd = 10
    cls_lmd = 1
    critic = 5

    # loading images on training
    domains = glob.glob(TRAIN_DIR+"/*")
    v_domains = glob.glob(VAL_DIR+"/*")

    num_domains = len(domains)
    assert num_domains == len(v_domains)

    btGen = BatchGenerator(img_size=img_size, imgdir=TRAIN_DIR, num_domains=num_domains)
    valGen = BatchGenerator(img_size=img_size, imgdir=VAL_DIR, num_domains=num_domains, aug=False)
    # sample images
    _Z = np.zeros([bs,img_size,img_size,3])
    _X, x_atr, _, y_atr, _, z_atr  = btGen.getBatch(bs)
    _Z = (_X + 1)*127.5
    #print(x_atr)
    #print(y_atr)
    #print(z_atr)
    _Z = tileImage(_Z)
    cv2.imwrite("input.png",_Z)


    #build models
    start = time.time()

    real_1 = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])
    real_2 = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])
    real_3 = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])
    label_1 = tf.placeholder(tf.float32, [bs, num_domains])
    label_2 = tf.placeholder(tf.float32, [bs, num_domains])
    label_3 = tf.placeholder(tf.float32, [bs, num_domains])
    v12 = label_2 - label_1
    alpha = tf.placeholder(tf.float32, [bs])
    rnd_ph = tf.placeholder(tf.float32, [])
    alpha_1 = alpha
    #alpha_t = tf.tile(alpha,[num_domains])
    alpha_t = tf.reshape(alpha, [bs,1])
    #alpha_t = tf.reshape(alpha_t, [bs,num_domains])
    #print(alpha)

    fake_12 = buildGenerator(real_1,v12,num_domains, reuse=False, name="gen")
    #print(fake_12)
    fake_11 = buildGenerator(real_1,label_2-label_2,num_domains, reuse=True, name="gen")
    fake_alp = buildGenerator(real_1,v12*alpha_t,num_domains, reuse=True, name="gen")

    g_loss = 0
    d_loss = 0
    adv_d_lambda = 1
    con_d_lambda = 10
    int_d_lambda = 10
    adv_g_lambda = 1
    con_g_lambda = 1
    int_g_lambda = 10
    cyc_g_lambda = 10
    sel_g_lambda = 10

    # Adversarial Loss(LS-GAN)
    adv_fake = buildDiscriminator(fake_12,None,v12,num_domains,reuse=[0,0],method="adv")
    adv_real = buildDiscriminator(real_2,None,v12,num_domains,reuse=[1,1],method="adv")
    d_adv_loss = (tf.reduce_mean(tf.square(adv_real-tf.ones_like(adv_real))) + tf.reduce_mean(tf.square(adv_fake-tf.zeros_like(adv_fake)))) * adv_d_lambda
    d_loss += d_adv_loss
    g_adv_loss =tf.reduce_mean(tf.square(adv_fake-tf.ones_like(adv_fake))) * adv_g_lambda
    g_loss += g_adv_loss

    # Interpolation Loss
    int_fake11 = buildDiscriminator(fake_11,None,0,num_domains,reuse=[1,0],method="int")
    int_fake12 = buildDiscriminator(fake_12,None,v12,num_domains,reuse=[1,1],method="int")
    int_fakealp = buildDiscriminator(fake_alp,None,v12*alpha_t,num_domains, reuse=[1,1],method="int")
    """
    d_int_loss = tf.reduce_mean(tf.where(rnd_ph==0,
        tf.square(int_fake11 - tf.zeros_like(int_fake11))+tf.square(int_fakealp - tf.reshape(alpha_1,[bs,1,1])),
        tf.square(int_fake12 - tf.zeros_like(int_fake12))+tf.square(int_fakealp -tf.reshape((1-alpha_1),[bs,1,1])))) * int_d_lambda
    """
    d_int_loss = tf.reduce_mean(tf.cond(tf.constant(rnd_ph==0, dtype=tf.bool),
        lambda:tf.square(int_fake11 - tf.zeros_like(int_fake11))+tf.square(int_fakealp - tf.reshape(alpha_1,[bs,1,1])),
        lambda:tf.square(int_fake12 - tf.zeros_like(int_fake12))+tf.square(int_fakealp -tf.reshape((1-alpha_1),[bs,1,1])))) * int_d_lambda

    d_loss += d_int_loss
    g_int_loss = tf.reduce_mean(tf.square(int_fakealp) - tf.zeros_like(int_fakealp)) * int_g_lambda
    g_loss += g_int_loss

    # Conditional Adversarial Loss
    v32 = label_2 - label_3
    v13 = label_3 - label_1

    sr = buildDiscriminator(real_1, real_2, v12, num_domains, reuse=[1,0],method="mat")
    sf = buildDiscriminator(real_1, fake_12, v12, num_domains, reuse=[1,1],method="mat")
    sw1 = buildDiscriminator(real_3, real_2, v12, num_domains, reuse=[1,1],method="mat")
    sw2 = buildDiscriminator(real_1, real_2, v32, num_domains, reuse=[1,1],method="mat")
    sw3 = buildDiscriminator(real_1, real_2, v13, num_domains, reuse=[1,1],method="mat")
    sw4 = buildDiscriminator(real_1, real_3, v12, num_domains, reuse=[1,1],method="mat")

    d_cond_loss = tf.reduce_mean(tf.square(sr - tf.ones_like(sr)))
    d_cond_loss += tf.reduce_mean(tf.square(sf - tf.zeros_like(sf)))
    d_cond_loss += tf.reduce_mean(tf.square(sw1 - tf.zeros_like(sw1)))
    d_cond_loss += tf.reduce_mean(tf.square(sw2 - tf.zeros_like(sw2)))
    d_cond_loss += tf.reduce_mean(tf.square(sw3 - tf.zeros_like(sw3)))
    d_cond_loss += tf.reduce_mean(tf.square(sw4 - tf.zeros_like(sw4)))
    d_cond_loss *= con_d_lambda
    d_loss += d_cond_loss
    g_cond_loss = tf.reduce_mean(tf.square(sf - tf.ones_like(sf))) * con_g_lambda
    g_loss += g_cond_loss

    # Cycle Reconstruction Loss
    fake_121 = buildGenerator(fake_12,-v12,num_domains, reuse=True, name="gen")
    g_rec_loss = tf.reduce_mean(tf.abs(real_1 - fake_121)) * cyc_g_lambda
    g_loss += g_rec_loss

    # Self Reconstruction Loss
    g_self_loss = tf.reduce_mean(tf.abs(real_1 - fake_11)) * sel_g_lambda
    g_loss += g_self_loss
    wd_gen = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="gen")
    wd_dis = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="dis")

    wd_gen = tf.reduce_sum(wd_gen)
    wd_dis = tf.reduce_sum(wd_dis)

    g_loss += wd_gen
    d_loss += wd_dis

    g_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(g_loss,
                    var_list=[x for x in tf.trainable_variables() if "gen" in x.name])
    d_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(d_loss,
                    var_list=[x for x in tf.trainable_variables() if "dis" in x.name])

    printParam(scope="gen")
    printParam(scope="dis")

    print("%.4e sec took building model"%(time.time()-start))

    start = time.time()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))

    sess =tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state('model')

    if ckpt: # checkpointがある場合
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)
        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")
    else:
        print("models were not found")
        init = tf.global_variables_initializer()
        #sess.run(init)

    print("%.4e sec took initializing"%(time.time()-start))

    gen_hist= []
    dis_hist= []

    start = time.time()

    for i in range(max_step + 1):

        x, x_label, y, y_label, z, z_label = btGen.getBatch(bs)
        x_labels = np.zeros([bs, num_domains])
        y_labels = np.zeros([bs, num_domains])
        z_labels = np.zeros([bs, num_domains])
        for b in range(bs):
            x_labels[b] = np.identity(num_domains)[x_label[b]]
            y_labels[b] = np.identity(num_domains)[y_label[b]]
            z_labels[b] = np.identity(num_domains)[z_label[b]]

        rnd = np.random.randint(2)
        alp = np.random.uniform(0, 0.5, size=bs) if rnd==0 else np.random.uniform(0.5, 1.0, size=bs)

        feed ={real_1:x, real_2:y, real_3:z, label_1:x_labels,
            label_2:y_labels, label_3:z_labels, alpha:alp, rnd_ph:rnd, lr: trans_lr}
        _, dis_loss, dis_adv, dis_cond, dis_int = sess.run(
            [d_opt, d_loss, d_adv_loss, d_cond_loss, d_int_loss], feed_dict=feed)

        rnd = np.random.randint(2)
        alp = np.random.uniform(0, 0.5, size=bs) if rnd==0 else np.random.uniform(0.5, 1.0, size=bs)
        feed ={real_1:x, label_1:x_labels, label_2:y_labels, alpha:alp, rnd_ph:rnd, lr: trans_lr}

        _, gen_loss, gen_adv, gen_cond, gen_int, gen_rec, gen_self = sess.run(
            [g_opt, g_loss, g_adv_loss, g_cond_loss, g_int_loss, g_rec_loss, g_self_loss], feed_dict=feed)

        trans_lr = trans_lr *(1 - 2/max_step)

        print("in step %s, dis_loss = %.4e,  gen_loss = %.4e"%(i, dis_loss, gen_loss))
        print("d_a=%.3e d_c=%.3e d_i=%.3e"%(dis_adv,dis_cond,dis_int))
        print("g_a=%.3e g_c=%.3e g_i=%.3e g_r=%.3e g_s=%.3e"%(gen_adv,gen_cond,gen_int,gen_rec,gen_self))

        dis_hist.append(dis_loss)
        gen_hist.append(gen_loss)

        if i % 100 ==0:
            x, x_label, y, y_label, z, z_label = valGen.getBatch(bs)
            x_labels = np.zeros([bs, num_domains])
            y_labels = np.zeros([bs, num_domains])
            z_labels = np.zeros([bs, num_domains])
            for b in range(bs):
                x_labels[b] = np.identity(num_domains)[x_label[b]]
                y_labels[b] = np.identity(num_domains)[y_label[b]]
                z_labels[b] = np.identity(num_domains)[z_label[b]]

            alp = np.random.uniform(0, 0.5, size=bs) if rnd==0 else np.random.uniform(0.5, 1.0, size=bs)
            feed ={real_1:x, label_1:x_labels, label_2:y_labels, alpha:alp, rnd_ph:rnd, lr: trans_lr}

            img_fake_A2B = sess.run(fake_12,feed_dict=feed)

            for im in range(len(x)):
                cv2.putText(x[im], '{}'.format(x_label[im]), (img_size-18, img_size-8), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), lineType=cv2.LINE_AA)
                cv2.putText(img_fake_A2B[im], '{}'.format(y_label[im]), (img_size-18, img_size-8), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), lineType=cv2.LINE_AA)
                cv2.putText(img_fake_A2B[im], '%.2f'%(alp[im]), (0, img_size-8), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), lineType=cv2.LINE_AA)

            _A = tileImage(x)
            _A2B = tileImage(img_fake_A2B)

            _Z = np.concatenate([_A,_A2B],axis=1)
            _Z = ( _Z + 1) * 127.5
            cv2.imwrite("%s/%s.png"%(SVIM_DIR, i),_Z)

            fig = plt.figure(figsize=(8,6), dpi=128)
            ax = fig.add_subplot(111)
            plt.title("Loss")
            plt.grid(which="both")
            plt.yscale("log")
            ax.plot(gen_hist,label="g_loss", linewidth = 0.25)
            ax.plot(dis_hist,label="d_loss", linewidth = 0.25)
            plt.xlabel('step', fontsize = 16)
            plt.ylabel('loss', fontsize = 16)
            plt.legend(loc='upper left')
            plt.savefig("histGAN.png")
            plt.close()

            print("%.4e sec took per100steps ,lr = %.4e" %(time.time()-start, trans_lr))
            start = time.time()

        if i%5000==0 and i!=0:
            saver.save(sess,os.path.join(SAVE_DIR,"model.ckpt"),i)
    sess.close()

if __name__ == '__main__':
    main()
