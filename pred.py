import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model import *

DATASET_DIR = "train"
VAL_DIR ="test"
MODEL_DIR = "model"
OUT_DIR_A2B = "out"
domains=os.listdir(DATASET_DIR)
num_domains = len(domains)

def main(arg):

    #folder_path = VAL_DIR + os.sep + arg[0]
    folder_path = arg.folder
    source_label = int(arg.source_label)
    target_label = arg.target_label
    alp = arg.interpolation
    img_size = arg.image_size

    #print("folderA = {}, direction = {} ".format(arg[0],domains[directions]))
    if not os.path.exists(OUT_DIR_A2B):
        os.makedirs(OUT_DIR_A2B)
    folderA2B = folder_path
    filesA2B = os.listdir(folderA2B)


    start = time.time()

    real_1 = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
    label_1 = tf.placeholder(tf.float32, [1, num_domains])
    label_2 = tf.placeholder(tf.float32, [1, num_domains])
    alpha = tf.placeholder(tf.float32, [])
    alpha_t = tf.reshape(alpha, [1,1])
    v12 = label_2 - label_1
    fake_alp = buildGenerator(real_1,v12*alpha_t,num_domains, reuse=False, name="gen",isTraining=False)

    sess = tf.Session()
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt: # checkpointがある場合
        last_model = ckpt.all_model_checkpoint_paths[1]
        #last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)

        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")

    else:
        print("checkpoints were not found.")
        print("saved model must exist in {}".format(MODEL_DIR))
        return

    print("%.4e sec took initializing"%(time.time()-start))

    start = time.time()
    #
    #print("{} has {} files".format(arg[0], len(filesA2B)))
    for i in range(len(filesA2B)):

        img_path = "{}/{}".format(folderA2B,filesA2B[i])
        img = cv2.imread(img_path)
        img = (img-127.5)/127.5
        h,w = img.shape[:2]

        img = cv2.resize(img,(img_size,img_size))
        input = img.reshape(1, img_size, img_size, 3)

        x_label = source_label
        y_label = target_label
        x_labels = np.zeros([1, num_domains])
        y_labels = np.zeros([1, num_domains])
        for b in range(1):
            x_labels[b] = np.identity(num_domains)[x_label]
            y_labels[b] = np.identity(num_domains)[y_label]

        feed ={real_1:input, label_1:x_labels, label_2:y_labels, alpha:alp,}

        out = sess.run(fake_alp,feed_dict=feed)
        out = out.reshape(img_size,img_size,3)
        out = np.concatenate([img, out],axis=1)
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        denorm_o = (out + 1) * 127.5
        cv2.imwrite(OUT_DIR_A2B+os.sep+'predicted_' + image_name + "_from" + str(source_label) +"to"+str(target_label)+ '.png', denorm_o)

    print("%.4e sec took for predicting" %(time.time()-start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--folder',"-f", dest='folder', type=str, default=None, help='folder name')
    parser.add_argument('--source_label',"-s", dest='source_label', type=int, default=None, help='source label')
    parser.add_argument('--target_label',"-t", dest='target_label', type=int, default=None, help='target label')
    parser.add_argument('--image_size',"-is", dest='image_size', type=int, default=256, help='image size')
    parser.add_argument('--interpolation',"-ip", dest='interpolation', type=float, default=1.0, help='interpolation late(0<interp<1)')
    args = parser.parse_args()

    main(args)
    """
    try:
        arg.append(sys.argv[1])
        arg.append(sys.argv[2])
        main(arg)
    except:
        print("Usage: python pred.py [folder] [source_label] [target_label] [interpolation]")
    """
