import glob
import cv2
import numpy as np
import os
import math
class BatchGenerator:
    def __init__(self, img_size, imgdir, num_domains, aug=True):
        self.rootPaths = glob.glob(imgdir+"/*")
        self.imgs_each_domain = []
        self.pathLens = []
        for i in range(num_domains):
            imagePath = glob.glob(self.rootPaths[i]+"/*")
            self.imgs_each_domain.append(imagePath)
            self.pathLens.append(len(imagePath))
        self.aug = aug
        self.imgSize = (img_size,img_size)
        self.labels = np.arange(num_domains)

        #self.direction = np.arange(num_domains)
        #self.direction = np.delete(self.direction,dirID)
        assert self.imgSize[0]==self.imgSize[1]

    def augment(self, img1):
        #軸反転 (inverse axis)
        if np.random.random() >0.5:
            img1 = cv2.flip(img1,1)

        """
        #軸移動 (moving axes)
        rand = (np.random.random()-0.5)/20
        y,x = img1.shape[:2]
        x_rate = x*(np.random.random()-0.5)/20
        y_rate = y*(np.random.random()-0.5)/20
        M = np.float32([[1,0,x_rate],[0,1,y_rate]])
        img1 = cv2.warpAffine(img1,M,(x,y),127)

        #回転 (rotate)
        rand = (np.random.random()-0.5)*5
        M = cv2.getRotationMatrix2D((x/2,y/2),rand,1)

        img1 = cv2.warpAffine(img1,M,(x,y))
        """
        return img1

    def getBatch(self,nBatch):
        x = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        y = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        z = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        x_atr = []
        y_atr = []
        z_atr = []

        for i in range(nBatch):
            labels = self.labels
            atr = np.random.choice(labels)
            img_path = np.random.choice(self.imgs_each_domain[atr])
            input = cv2.imread(img_path)
            input = cv2.resize(input,self.imgSize)
            if self.aug:
                input = self.augment(input)
            x[i,:,:,:] = (input - 127.5) / 127.5
            x_atr.append(atr)

            labels = labels[labels!=atr]
            atr = np.random.choice(labels)
            img_path = np.random.choice(self.imgs_each_domain[atr])
            input = cv2.imread(img_path)
            input = cv2.resize(input,self.imgSize)
            if self.aug:
                input = self.augment(input)
            y[i,:,:,:] = (input - 127.5) / 127.5
            y_atr.append(atr)

            labels = labels[labels!=atr]
            atr = np.random.choice(labels)
            img_path = np.random.choice(self.imgs_each_domain[atr])
            input = cv2.imread(img_path)
            input = cv2.resize(input,self.imgSize)
            if self.aug:
                input = self.augment(input)
            z[i,:,:,:] = (input - 127.5) / 127.5
            z_atr.append(atr)

        return x, x_atr, y, y_atr, z, z_atr

if __name__ == '__main__':
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

    TRAIN_DIR = "train"
    domains = os.listdir(TRAIN_DIR)
    num_domains = len(domains)
    img_size = 128
    bs = 16
    btGen = BatchGenerator(img_size=img_size, imgdir=TRAIN_DIR, num_domains=num_domains)

    # sample images
    _Z = np.zeros([num_domains,img_size,img_size,3])

    _X, x_atr, _, _, _, _  = btGen.getBatch(bs)
    _Z = (_X + 1)*127.5
    print(x_atr)
    _Z = tileImage(_Z)
    cv2.imwrite("input.png",_Z)
