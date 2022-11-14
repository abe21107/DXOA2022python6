
import os, pickle, cv2
import numpy as np
from util import color
import pandas as pd

class dataset:
    def __init__(self, path="_tmp/results.pkl"):
        with open("_tmp/results.pkl","rb") as f:
            tmp = pickle.load(f)
        self.images = tmp["images"]
        self.boxes  = tmp[ "boxes"]
        self.ntim = self.boxes.shape[0]
        self.nind = self.boxes.shape[1]
        self.color = color

        self.img_with_bbox = self.getimg()
        self.y = self.count()
        #print(self.y)

    def dump(self,n):
        img   = self.images[n]
        boxes = self.boxes[n]
        ind = np.where(np.isnan(boxes[:,0])==0)[0]
        boxes =      boxes[ind].astype(np.int64)
        for i,box in zip(ind,boxes):
            img = cv2.rectangle(img, pt1=tuple(box[0:2]), pt2=tuple(box[2:4]), color=self.color[int(i)], thickness=2)
        return img

    def getimg(self):
        o = []
        for n in range(self.ntim):
            o.append(self.dump(n))
        return np.array(o)

    def count(self, x0=None, x1=None, y0=None, y1=None):
        if x0 is None:
            boxes = np.sum(1.0-np.isnan(self.boxes[:,:,0]),axis=-1)
        else:
            nt,nb,ch = self.boxes.shape
            boxes = self.boxes.reshape([-1,ch])
            ind = np.where(np.isnan(boxes[:,0])==False)[0]
            boxes = boxes[ind]
            ix0 = np.where(boxes[:,0]>x0,1,0)
            iy0 = np.where(boxes[:,1]>y0,1,0)
            ix1 = np.where(boxes[:,2]<x1,1,0)
            iy1 = np.where(boxes[:,3]<y1,1,0)

            out = np.zeros([nt*nb], dtype=np.float32)
            out[ind] = ix0*iy0*ix1*iy1
            out = out.reshape([nt,nb])
            boxes = np.sum(out,axis=-1)
        return pd.DataFrame(boxes)

if __name__ == '__main__':
    d = dataset("_tmp/results.pkl")
    y = d.count(0,100,0,100)
    print(y)
