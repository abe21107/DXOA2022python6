
import os, cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models  import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils   import draw_outputs
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class yolo:
    def __init__(self, classes = './checkpoints/coco.names',
                       weights = './checkpoints/yolov3.tf',
                       size = 416):
        self.size = size
        self.class_names = [c.strip() for c in open(classes).readlines()]
        self.num_classes = len(self.class_names)
        print('[ok] classes loaded', self.num_classes)
        self.model = YoloV3(classes=self.num_classes)
        print('[ok] yolo model defined')
        self.model.load_weights(weights).expect_partial()
        print('[ok] weights loaded')

    # input: numpy array [h,w,ch], ch is BGR, value[0-255]uint8
    @tf.function()
    def pred(self, image):
        img = tf.expand_dims(image, 0)
        img = transform_images(img, self.size)
        boxes, scores, classes, nums = self.model(img)
        return boxes, scores, classes, nums

    def pred_and_draw(self, image):
        boxes, scores, classes, nums = self.pred(image)
        return draw_outputs(image, (boxes, scores, classes, nums), self.class_names)

if __name__ == '__main__':
    m = yolo()
    img = cv2.imread("../_image/street.jpg")
    out = m.pred_and_draw(img)
    cv2.imwrite("model.jpg", out)
