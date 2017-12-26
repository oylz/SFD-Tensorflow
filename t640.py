

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_640, np_methods
from util import getter 
import cv2
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm




# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

image_pre = getter.TreateImage(img_input)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
#reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_640.SSDNet()

print("+++++++++++++++++++++++++++++++++++++")
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    #predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False)

print("~~~~~~~~predictions:", predictions)
print("~~~~~~~~localisations:", localisations)
print("======================mmmmmmmmmmmmmmmmmmmmm==========")
# Restore SSD model.
ckpt_filename = './tt/s3fd/SFD_fast_v2.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
net_shape = getter.EVAL_SIZE
ssd_anchors = ssd_net.anchors(net_shape)





# Main image processing routine.
def process_image(img):
    # Run SSD network.
    
    rpredictions, rlocalisations = isess.run(
                            [predictions, localisations],
                            feed_dict={img_input: img})

    # TreateBoxes
    rclasses, rscores, rbboxes = np_methods.TreateBoxes(
                       rpredictions, 
                       rlocalisations, 
                       ssd_anchors,
                       decode=True)
    print("[[[[rbboxes shape:", rbboxes.shape, "]]]]") 
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, 
                                            rscores, 
                                            rbboxes, 
                                            top_k=5000)
    #rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, 
    #                                        rscores, 
    #                                        rbboxes, 
    #                                        nms_threshold=0.45)#nms_threshold)
    return rclasses, rscores, rbboxes



def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            #if cls_id!=1 or score<0.6:
            #if score<0.6:
            if score<0.6:
                continue

            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            print("drawrect(", xmin, ymin, xmax, ymax, ")")
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()




# Test on some demo image and visualize output.
path = './demo/'
image_names = sorted(os.listdir(path))

#pp =  path + image_names[-5]
#pp = "/home/xyz/code1/xyz2/img1/000380.jpg"
pp = "/home/xyz/code1/xyz/img1/000417.jpg"

print("=======[[[[[[" + pp + "]]]]]]========")
img = mpimg.imread(pp)

#dimg = getter.TreateImage(img)
rclasses, rscores, rbboxes =  process_image(img)

print("***************rbboxes[", rbboxes, "]*****************")
plt_bboxes(img, rclasses, rscores, rbboxes)







