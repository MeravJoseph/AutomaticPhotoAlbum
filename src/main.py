import numpy as np
import tensorflow as tf

from inception.inception_v4 import inception_v4
from inception import inception_utils
slim = tf.contrib.slim

def run_model():
    batch_size = 5
    height, width = 299, 299

    checkpoint_path = r"C:\Users\meravj\workspace\repos\AutomaticPhotoAlbum\trained\inception_v4.ckpt"

    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        arg_scope = inception_utils.inception_arg_scope()

        # eval_inputs = tf.random_uniform((batch_size, height, width, 3))
        inputs = tf.placeholder(tf.float32, (None, height, width, 3))

        with slim.arg_scope(arg_scope):
            logits, end_points = inception_v4(inputs, is_training=False)
        predictions = tf.argmax(logits, 1)

        # Create a saver.
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)

            # output = sess.run(predictions)
            for batch_num in range(1):
                images = np.ones((batch_size, height, width, 3), dtype=np.uint8)
                descriptor = sess.run(end_points['PreLogitsFlatten'], feed_dict={inputs: images})
                print(descriptor)


if __name__ == "__main__":
    run_model()