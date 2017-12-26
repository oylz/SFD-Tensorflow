import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers

slim = tf.contrib.slim


# only used in flow function:AnchorAllLayers(), final for outapp to detect
def AnchorOneLayer(img_shape,
                         feat_shape,
                         anchor_size,
                         anchor_step,
                         anchor_offset,
                         dtype):
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + anchor_offset) * anchor_step / img_shape[0]
    x = (x.astype(dtype) + anchor_offset) * anchor_step / img_shape[1]

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    num_anchors = 1 
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    print("onelayer0----anchor_size", anchor_size)
    print("onelayer1----x.shape:", x.shape, ", y.shape:", y.shape , ", w.shape:", w.shape, ", h.shape:", h.shape)

    h[0] = anchor_size / img_shape[0]
    w[0] = anchor_size / img_shape[1]

    print("onelayer2----x.shape:", x.shape, ", y.shape:", y.shape, ", w.shape:", w.shape, ", h.shape:", h.shape)
    print("onelayer3----w:", w, ", h:", h)
    return y, x, h, w

# only used in SSDNet.anchors(),final for outapp to detect
def AnchorAllLayers(img_shape,
                           feat_shapes,
                           anchor_sizes,
                           anchor_steps,
                           anchor_offset,
                           dtype):
    layers_anchors = []
    for i, feat_shape in enumerate(feat_shapes):
        anchor_bboxes = AnchorOneLayer(img_shape, 
                                             feat_shape,
                                             anchor_sizes[i],
                                             anchor_steps[i],
                                             anchor_offset=anchor_offset, 
                                             dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# only used in SSDNet.net()
def FeatShapesFromNet(predictions, default_shapes):
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes

# only used in MultiboxLayer
def tensor_shape(x, rank=3):
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

# only used in ssd_net
def MultiboxLayer(addn, 
                       inputs,
                       num_classes,
                       normalization):
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, 
                  scaling=True)
    print("nnnn------------begin MultiboxLayer----------nnnn")

    num_loc_pred = 4 # [4] no used if caffe convert.
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], 
                           activation_fn=None,
                           scope='conv_loc')
    print("====loc_pred0:", loc_pred)

    loc_pred = custom_layers.channel_to_last(loc_pred)
    print("====loc_pred1:", loc_pred)
    tt0 = tensor_shape(loc_pred, 4)
    print("====tt0:", tt0)
    print("====tt0[:-1]:", tt0[:-1])
    num_anchors = 1 
    loc_pred = tf.reshape(loc_pred,
                          tt0[:-1]+[num_anchors, -1])


    print("====loc_pred2:", loc_pred)

    # Class prediction.
    num_cls_pred = (num_anchors+addn)*2 #no used if caffe convert
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], 
                           activation_fn=None,
                           scope='conv_cls')
    print("====num_cls_pred:", num_cls_pred, ", cls_pred0:", cls_pred)

    cls_pred = custom_layers.channel_to_last(cls_pred)
    print("====cls_pred1:", cls_pred)

    tt = tensor_shape(cls_pred, 4)
    print("====tt:", tt)
    cls_pred = tf.reshape(cls_pred,
                          tt[:-1]+[num_anchors, -1])
    print("====cls_pred2:", cls_pred)
    print("uuuu------------end   MultiboxLayer----------uuuu")
    return cls_pred, loc_pred

# only used in SSDNet.net(), and extend for nets_factory
def ssd_net(inputs,
            num_classes,
            feat_layers,
            normalizations,
            is_training,
            dropout_keep_prob,
            prediction_fn,
            reuse,
            scope):
    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_640_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        print("nnnn-block1 begin")
        net = slim.repeat(inputs, 2, slim.conv2d, 16, [3, 3], scope='r2_crcr1', 
                         activation_fn=tf.nn.relu)
        end_points['block1'] = net
        print("uuuu-block1 end")

        print("nnnn-block2 begin")
        net = slim.max_pool2d(net, [2, 2], scope='bbpool1',
                         stride=2)
        net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='r2_crcr2', 
                         activation_fn=tf.nn.relu)
        end_points['block2'] = net
        print("uuuu-block2 end")


        print("nnnn-block3 begin")
        net = slim.max_pool2d(net, [2, 2], scope='ddpool2',
                         stride=2)
        net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='r3_crcr3', 
                         activation_fn=tf.nn.relu)
        end_points['block3'] = net
        print("uuuu-block3 end")


        print("nnnn-block4 begin")
        net = slim.max_pool2d(net, [2, 2], scope='ffpool3',
                         stride=2)
        net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='r3_crcr4', 
                        activation_fn=tf.nn.relu)
        end_points['block4'] = net
        print("uuuu-block4 end")


        print("nnnn-block5 begin")
        net = slim.max_pool2d(net, [2, 2], scope='hhpool4',
                         stride=2)
        # rate as `[dilation]`/`pad` in prototxt?, if is `[dilation]` then set rate=1
        net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], rate=1, scope='r3_crcr5', 
                        activation_fn=tf.nn.relu)
        end_points['block5'] = net
        print("uuuu-block5 end")


        print("nnnn-block6 begin")
        # pool5: kernel_size: 3->2, stride: 1->2, +pad:1,, where to put `pad:1`?
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='jjpool5')
        net = slim.conv2d(net, 256, [3, 3], scope='kkfc6',
                        activation_fn=tf.nn.relu)
        end_points['block6'] = net
        print("uuuu-block6 end")


        print("nnnn-block7 begin")
        #oylzoylzoylz net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        net = slim.conv2d(net, 256, [1, 1], scope='llfc7', 
                        activation_fn=tf.nn.relu)
        end_points['block7'] = net
        print("uuuu-block7 end")


        print("nnnn-block8 begin")
        # conv61->conv62
        #oylzoylzoylz net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        end_point = 'block8'
        with tf.variable_scope(end_point):
            # paper: 1x1x128
            net = slim.conv2d(net, 128, [1, 1], scope='mmconv1x1', 
                        activation_fn=tf.nn.relu)
            net = custom_layers.pad2d(net, pad=(1, 1))
            # paper: 3x3x512-s2
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='nnconv3x3', padding='VALID', 
                        activation_fn=tf.nn.relu)
        end_points[end_point] = net
        print("uuuu-block8 end")

        print("nnnn-block9 begin")
        end_point = 'block9'
        # conv71->conv72
        with tf.variable_scope(end_point):
            # paper: 1x1x128
            net = slim.conv2d(net, 128, [1, 1], scope='ooconv1x1',
                        activation_fn=tf.nn.relu)
            net = custom_layers.pad2d(net, pad=(1, 1))
            # paper: 3x3x256-s2
            net = slim.conv2d(net, 128, [3, 3], stride=2, scope='ppconv3x3', padding='VALID',
                        activation_fn=tf.nn.relu)
        end_points[end_point] = net
        print("uuuu-block9 end")


        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        addn = 1
#atation, in `process block3` 3 layers more than ori ssd-300 prototxt:
#  [
#  name: "conv3_3_norm_mbox_conf_maxout"
#  name: "conv3_3_norm_mbox_conf_maxout_slice"
#  name: "conv3_3_bg_maxout"
#  ]
#
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                print("nnnn-begin process----" + layer + '_box')
                p, l = MultiboxLayer(addn, 
                                          end_points[layer],
                                          num_classes,
                                          normalizations[i])
                addn = 0
                print("uuuu-end process----" + layer + '_box')

            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)
        print("[final end]")
        return predictions, localisations, logits, end_points



#==========================================================
#used in SSDNet, and extend for nets_factory
def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


#used in SSDNet, and extend for nets_factory
def ssd_arg_scope_caffe(caffe_scope):
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc


