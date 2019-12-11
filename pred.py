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
    folder_path = arg[0]
    directions = int(arg[1])

    print("folderA = {}, direction = {} ".format(arg[0],domains[directions]))
    if not os.path.exists(OUT_DIR_A2B):
        os.makedirs(OUT_DIR_A2B)
    folderA2B = folder_path
    filesA2B = os.listdir(folderA2B)
    img_size = 256

    start = time.time()

    real_A = tf.placeholder(tf.float32, [1, img_size, img_size, 3 ])
    label_A2B = tf.placeholder(tf.float32, [1, num_domains]) # target image atr
    fake_A2B = buildGenerator(real_A,label_A2B,num_domains, reuse=False, nBatch=1, name="gen")

    sess = tf.Session()
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt: # checkpointがある場合
        #last_model = ckpt.all_model_checkpoint_paths[3]
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
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
    print("{} has {} files".format(arg[0], len(filesA2B)))
    for i in range(len(filesA2B)):

        img_path = "{}/{}".format(folderA2B,filesA2B[i])
        img = cv2.imread(img_path)
        img = (img-127.5)/127.5
        h,w = img.shape[:2]

        input = cv2.resize(img,(img_size,img_size))
        input = input.reshape(1, img_size, img_size, 3)

    
        directions = np.vectorize(int)(directions)
        one_hot_dir = np.identity(num_domains)[directions]
        one_hot_dir = one_hot_dir.reshape(1,num_domains)

        out = sess.run(fake_A2B,feed_dict={real_A:input, label_A2B:one_hot_dir})
        out = out.reshape(img_size,img_size,3)
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        denorm_o = (out + 1) * 127.5
        cv2.imwrite(OUT_DIR_A2B+os.sep+'predicted_' + image_name + "_" + str(directions) + '.png', denorm_o)

    print("%.4e sec took for predicting" %(time.time()-start))

if __name__ == '__main__':
    arg = []
    try:
        arg.append(sys.argv[1])
        arg.append(sys.argv[2])
        main(arg)
    except:
        print("Usage: python pred.py [folder] [direction]")
