
from enum import Enum, IntEnum
import numpy as np

import tensorflow as tf
import tf_extended as tfe

from tensorflow.python.ops import control_flow_ops

from util import tf_image

slim = tf.contrib.slim

# Resizing strategies.
Resize = IntEnum('Resize', ('NONE',                # Nothing!
                            'CENTRAL_CROP',        # Crop (and pad if necessary).
                            'PAD_AND_RESIZE',      # Pad, and resize to output shape.
                            'WARP_RESIZE'))        # Warp resize.

# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

# Some training pre-processing parameters.
BBOX_CROP_OVERLAP = 0.5         # Minimum overlap to keep a bbox after cropping.
MIN_OBJECT_COVERED = 0.25
CROP_RATIO_RANGE = (0.6, 1.67)  # Distortion ratio during cropping.
EVAL_SIZE = (640, 640)


def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image



def TreateImage(image, data_format='NHWC', scope='ssd_preprocessing_train'):
    with tf.name_scope(scope):
        image = tf.to_float(image)
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])

	out_shape=EVAL_SIZE	
        image = tf_image.resize_image(image, out_shape,
                                      method=tf.image.ResizeMethod.BILINEAR,
                                      align_corners=False)

        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image



