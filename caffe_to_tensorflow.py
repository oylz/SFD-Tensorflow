

import tensorflow as tf

from nets import caffe_scope
from nets import ssd_vgg_640

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'caffemodel_path', None,
    'The path to the Caffe model file to convert.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    # Caffe scope...
    caffemodel = caffe_scope.CaffeScope()
    caffemodel.load(FLAGS.caffemodel_path)

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        global_step = slim.create_global_step()

        ssd_net = ssd_vgg_640.SSDNet()
        ssd_shape = ssd_net.params.img_shape

        # Image placeholder and model.
        shape = (1, ssd_shape[0], ssd_shape[1], 3)
        img_input = tf.placeholder(shape=shape, dtype=tf.float32)
        # Create model.
        with slim.arg_scope(ssd_net.arg_scope_caffe(caffemodel)):
            ssd_net.net(img_input, is_training=False)

        init_op = tf.global_variables_initializer()
        with tf.Session() as session:
            # Run the init operation.
            session.run(init_op)

            # Save model in checkpoint.
            saver = tf.train.Saver()
            ckpt_path = FLAGS.caffemodel_path.replace('.caffemodel', '.ckpt')
            saver.save(session, ckpt_path, write_meta_graph=False)


if __name__ == '__main__':
    tf.app.run()

