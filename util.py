
import sys, json, cv2
import numpy as np
from model import yolo

def tocsv(path, data):
    with open(path,"w") as f:
        for d in data:
            line = ""
            for _d in d: line += "{},".format(_d)
            f.write(line[:-1]+"\n")

class progress:
    def __init__(self, nmax=None, MAX_LEN=50):
        self.nmax    =    nmax
        self.MAX_LEN = MAX_LEN
    def get_progressbar_str(self, progress):
        BAR_LEN = int(self.MAX_LEN*progress)
        return ('[' + '=' * BAR_LEN +
                ('>' if BAR_LEN < 100 else '') +
                ' ' * (self.MAX_LEN - BAR_LEN) +
                '] %.1f%%' % (progress * 100.))
    def run(self, progress):
        if self.nmax is None:
            sys.stderr.write('\r\033[K' + self.get_progressbar_str(progress))
        else:
            sys.stderr.write('\r\033[K' + self.get_progressbar_str(progress/self.nmax))
        sys.stderr.flush()
    def end(self):
        sys.stderr.write('\r\033[K' + self.get_progressbar_str(1.0))
        sys.stderr.flush()
        sys.stderr.write('\n')
        sys.stderr.flush()

######
###### optical-flow
######
class control:
    def __init__(self, path):
        with open(path) as f:
            self.config = json.load(f)
        self.vmax = len(self.config)
        self.set()
        self.of0 = 0.5
        self.of1 =   3
        self.of2 =  15
        self.of3 =   3
        self.of4 =   5
        self.of5 = 1.2
        self.of6 =   0

    def set(self, n=0):
        self.path = self.config[n]["path"]
        self.skip = self.config[n]["skip"]
        self.cap  = cv2.VideoCapture(self.path) # 動画ファイルの読み込み
        self.h    = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 動画の縦のピクセル数
        self.w    = int(self.cap.get( cv2.CAP_PROP_FRAME_WIDTH)) # 動画の横のピクセル数
        self.fps  = self.cap.get(cv2.CAP_PROP_FPS)               # 動画のFPS（1秒に何フレームか）
        self.fmax = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画の総フレーム数
        self.nmax = self.fmax//self.skip

        ret, frame = self.cap.read()
        self.prvs = frame
        self.cunt = 0
        print("path:", self.path, self.nmax)
        print(" h/w:", self.h, self.w)
        print(" fps:", self.fps)
        self.hsv  = np.zeros([self.h,self.w,3], dtype=np.uint8)
        self.hsv[...,1] = 255

    def getnext(self):
        for n in range(self.skip):
            self.ret, self.next = self.cap.read()
            self.cunt += 1

    def update(self):
        self.prvs = self.next

    def of(self, prvs=None, next=None, is_rgb=False):
        if prvs is None: prvs = cv2.cvtColor(self.prvs,cv2.COLOR_BGR2GRAY)
        if next is None: next = cv2.cvtColor(self.next,cv2.COLOR_BGR2GRAY)
        # http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None,
                                            self.of0,
                                            self.of1,
                                            self.of2,
                                            self.of3,
                                            self.of4,
                                            self.of5,
                                            self.of6)
        rgb = None
        if is_rgb:
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            self.hsv[...,0] = ang*180/np.pi/2
            self.hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(self.hsv,cv2.COLOR_HSV2BGR)
        return flow, rgb

    def dumpof(self, root="_tmp/{:04d}.png"):
        prvs = cv2.cvtColor(self.prvs,cv2.COLOR_BGR2GRAY)
        for n in range(1,self.nmax):
            self.getnext()

            flow,rgb = self.of(cv2.cvtColor(self.prvs,cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(self.next,cv2.COLOR_BGR2GRAY),True)
            cv2.imwrite(root.format(n), np.concatenate([self.next,rgb],axis=0))

            self.update()
            if n%10 == 0: print(n, self.nmax)

######
######
######
class summary:
    def __init__(self, boxes, iou_limit=0.5, is_clean=True, is_check=False):
        self.boxes = boxes
        self.nmax  = len(boxes)

        self.done = None
        self.data = np.array(self.boxes[0][-1], dtype=np.float32)[np.newaxis]
        ind = np.arange(self.data.shape[1])
        for n in range(1,self.nmax):
            add,b2a,score = self.iou(n)
            new = self.data[:,b2a]
            ind = np.where(score<iou_limit)[0]
            new[:,ind] = np.nan

            done = self.remove(b2a,score,iou_limit)
            if done is not None: self.done = self.concat(self.done, done, axis=1)

            self.data = np.concatenate([new,add[np.newaxis]], axis=0)
            if is_check: print(n, self.data.shape, self.done.shape)
            #break

        if self.data.shape[0] == self.nmax: self.done = self.concat(self.done, self.data, axis=1)
        if is_clean: self.clean()
        if is_check: print("done:", self.done.shape)

    def concat(self, a, b, axis=0):
        if a is None:
            return b
        else:
            return np.concatenate([a,b],axis=axis)

    def remove(self, b2a, score, iou_limit):
        o = -np.ones([self.data.shape[1]])
        i = np.where(score>=iou_limit)[0]
        b2a = b2a[i]
        o[b2a] = 1
        i = np.where(o<0)[0]
        if i.shape[0] > 1:
            o = np.zeros([self.nmax,i.shape[0],self.data.shape[2]])
            o[:,:,:] = np.nan
            o[:self.data.shape[0]] = self.data[:,i]
            return o
        else:
            return None

    def clean(self):
        isnan = np.sum(1.0 - np.isnan(self.done[:,:,0]), axis=0)
        ind = np.where(isnan>1)[0]
        self.done = self.done[:,ind]

    def iou(self, n, is_check=False):
        A = np.array(self.boxes[n][0]) #(N,4)
        B = np.array(self.boxes[n][1]) #(M,4)
        A = np.clip(A,0,1e+4)
        B = np.clip(B,0,1e+4)
        N = A.shape[0]
        M = B.shape[0]
        #(N,M)
        a = np.repeat(A[:,np.newaxis], M, axis=1)
        b = np.repeat(  B[np.newaxis], N, axis=0)
        area_a = (a[:,:,2]-a[:,:,0]+1)*(a[:,:,3]-a[:,:,1]+1)
        area_b = (b[:,:,2]-b[:,:,0]+1)*(b[:,:,3]-b[:,:,1]+1)
        w = np.maximum(0, np.minimum(a[:,:,2], b[:,:,2]) - np.maximum(a[:,:,0], b[:,:,0]) + 1)
        h = np.maximum(0, np.minimum(a[:,:,3], b[:,:,3]) - np.maximum(a[:,:,1], b[:,:,1]) + 1)
        area_i = w*h
        iou = area_i/(area_a + area_b - area_i)
        #(M)
        b2a = np.argmax(iou, axis=0)
        scr =    np.max(iou, axis=0)
        if is_check:
            out = A[b2a]
            ind = np.where(scr<0.5)[0]
            out[ind] = np.nan
            for a,b in zip(out,B):
                print(a,b)
        return B,b2a,scr

######
######
######
class yolof:
    def __init__(self, path="config.json"):
        self.m =    yolo()     ; print("define yolo done!")
        self.v = control(path) ; print("define video capture done!")
        self.whwh = np.array([self.v.w,self.v.h,self.v.w,self.v.h], dtype=np.int32)

    def xyxy(self, img):
        boxes, objectness, classes, nums = self.m.pred(img)
        o = []
        for box in boxes[0,:nums[0]]:
            o.append((np.array(box[0:4]) * self.whwh).astype(np.int32))
        return np.array(o), classes[0,:nums[0]]

    def shift(self,boxes,flow):
        new = []
        for box in boxes:
            h0 = np.clip(box[1],0,self.v.h-1)
            h1 = np.clip(box[3],0,self.v.h-1)
            w0 = np.clip(box[0],0,self.v.w-1)
            w1 = np.clip(box[2],0,self.v.w-1)
            f  = flow[h0:h1, w0:w1]
            f  = f.reshape([-1,2])
            f  = np.mean(f, axis=0)
            ff = np.concatenate([f,f],axis=0)
            new.append( box+ff )
        new = np.array(new)
        return new.astype(np.int32)

    def drawbox(self, bBs=None, bGs=None, bRs=None):
        if bBs is not None:
            for box in bBs:
                img = cv2.rectangle(img, tuple(box[0:2]), tuple(box[2:4]), (255,0,0), 2)
        if bGs is not None:
            for box in bGs:
                img = cv2.rectangle(img, tuple(box[0:2]), tuple(box[2:4]), (0,255,0), 2)
        if bRs is not None:
            for box in bRs:
                img = cv2.rectangle(img, tuple(box[0:2]), tuple(box[2:4]), (0,0,255), 2)
        return img

    def flow(self, root=None):
        p = progress(self.v.nmax)
        images  = []
        boxes   = []
        classes = []
        b_prvs, c_prvs = self.xyxy(self.v.prvs)
        images.append(self.v.prvs.copy())
        boxes.append([None,b_prvs])
        classes.append(c_prvs.numpy())
        for n in range(1,self.v.nmax):
            self.v.getnext()
            b_next, c_next = self.xyxy(self.v.next)
            flow, rgb = self.v.of(is_rgb=True)
            b_pred = self.shift(b_prvs, flow)

            images.append(self.v.next.copy())
            #boxes.append([b_next,b_prvs])
            boxes.append([b_pred,b_next])
            classes.append(c_prvs.numpy())

            if root is not None:
                im0 = self.drawbox(self.v.next.copy(), b_prvs, None, b_next)
                im1 = self.drawbox(self.v.next.copy(), b_pred, None, b_next)
                img = np.concatenate([im0,im1,rgb],axis=0)
                cv2.imwrite(root.format(n), img)

            self.v.update()
            b_prvs, c_prvs = b_next, c_next
            p.run(n)
            #break
        p.end()
        images = np.array(images)
        images = images.astype(np.uint8)
        return images, boxes, classes

######
######
######
color = [(234, 191,  48),
         (233, 153,  38),
         (167, 225, 100),
         (216, 163, 149),
         (152, 100, 234),
         (222, 144, 171),
         (178,  34, 248),
         (192,  22, 105),
         (172,  48, 114),
         (  4,  62,  95),
         (  4,   0,  25),
         (119,  99, 207),
         ( 77,  19,  30),
         ( 34, 150, 106),
         (222,  44, 206),
         (231,  62, 241),
         ( 46,  15, 237),
         (129, 142, 182),
         (180, 233, 173),
         ( 62,  67, 108),
         (200, 157, 148),
         (163, 192, 181),
         (142, 248, 173),
         ( 29, 166, 223),
         (133,  36, 154),
         ( 41,   6, 192),
         (197,  36, 228),
         ( 43, 180, 114),
         (210, 147,  48),
         (  1, 171, 207),
         ( 76, 195, 148),
         (139,  78,  34),
         (170,  46, 187),
         (200,  73, 108),
         (194, 121, 185),
         ( 24, 216, 152),
         ( 23, 108, 197),
         (138,  51,  46),
         ( 12,  32, 183),
         (240,  49, 115),
         (183,  43, 147),
         ( 97,  78,  97),
         (219, 169,  98),
         (110, 166, 200),
         (151, 146,  93),
         ( 28,  56,  99),
         (232, 184, 161),
         ( 72, 178, 248),
         (184,  16, 148),
         (238, 130, 185),
         ( 75,  62,  79),
         ( 10, 148, 161),
         ( 51,  84, 197),
         ( 65,  82, 186),
         ( 60,  29,  50),
         (210,  87, 102),
         (113,   9,  10),
         (229, 106,  40),
         (170, 188, 188),
         (205, 149, 198),
         (155,  56, 252),
         ( 77,  48, 192),
         (227, 251, 176),
         (115, 143, 185),
         (239,  78, 112),
         ( 39, 143, 177),
         ( 57, 100,  38),
         (232, 114, 175),
         (142, 240, 235),
         (251, 175,  64),
         (100, 211,  91),
         (174,   1, 235),
         (249,  78,  63),
         (133, 251, 253),
         ( 23, 254,  58),
         ( 20, 158,  79),
         (225,  51,  39),
         ( 68, 171, 118),
         (214, 213, 117),
         ( 89,  75, 213),
         ( 55, 165,  24),
         (174, 120, 239),
         ( 55, 158, 150),
         (  9, 249,  88),
         ( 32,  21, 230),
         (206,  11,  97),
         ( 82, 132, 109),
         (143, 207, 125),
         (138, 207, 155),
         ( 49, 193, 152),
         ( 10,  65, 125),
         (225,  62, 104),
         (133,  48,  66),
         ( 56,  63,  97),
         (217,  12,  51),
         ( 72, 147, 197),
         ( 41, 234, 234),
         (190, 137,  32),
         (179, 161, 230),
         ( 14,   7,  87),
         ( 81,  65, 237),
         (158,  50, 108),
         (171, 109, 241),
         (  9, 125,   8),
         ( 48,  40, 237),
         (223,  60, 117),
         (181, 218, 214),
         ( 26,  84, 146),
         (105, 115,  47)]

if __name__ == "__main__":
    #c = control("config.json")
    #c.dumpof()
    pass

