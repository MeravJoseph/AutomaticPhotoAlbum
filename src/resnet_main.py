import numpy as np
import tensorflow as tf
import cv2
import os
import sys
import itertools

from resnet.resnet_v2 import resnet_v2_50
from resnet import resnet_utils
from sklearn.cluster import KMeans
import numpy as np

slim = tf.contrib.slim

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)


def get_data_list(dir_path, size):
    """

    :param dir_path: images directory path
    :param size: image target size 
    :return: list of images, in a uniform size of size x size, anda list of their full path
    """

    images = os.listdir(dir_path)
    img_list = []
    img_path_list = []
    for fn in images:
        cur_fn = os.path.abspath(os.path.join(dir_path, fn))
        img = cv2.imread(cur_fn, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 3:
            img_list.append(fit_image(img, size))
            img_path_list.append(cur_fn)
    return img_list, img_path_list


def fit_image(rgb_img, size):
    """
    given an RGB image, return the new image in the given size, 
    using crop from the center and resize

    :param img: the image that should be resized
    :param size: image target size 
    :return: image size of size x size
    """
    size_x = rgb_img.shape[1]
    size_y = rgb_img.shape[0]
    center = (np.uint32(size_y / 2), np.uint32(size_x / 2))

    if size_x < size_y:
        cropped = rgb_img[(center[0] - center[1]):(center[0] + center[1] + (size_x % 2)), :]
    else:
        cropped = rgb_img[:, (center[1] - center[0]):(center[0] + center[1] + (size_x % 2))]

    resized = cv2.resize(cropped, (size, size))
    return resized


def create_image_batches(img_list, img_path_list, batch_size):
    """

    :param img_list: list of images
    :param batch_size: size of wanted batch
    :return: list of batches of images, where each batch maximum size is batch_size
    """
    batch_num = np.uint32(np.ceil(len(img_list) / batch_size))
    batches = []
    paths_batches = []
    for i in range(batch_num):
        batches.append(img_list[i * batch_size: ((i + 1) * batch_size)])
        paths_batches.append(img_path_list[i * batch_size: ((i + 1) * batch_size)])
    return batches, paths_batches


def run_model(img_list):
    """
    Gets image batches, run the pretrained inception_v4 on all the
    images, and return batches of descriptors (one for each image)
    :param batches: image batches 
    :return: descriptor batches
    """
    batch_size = 1
    height, width = 224, 224
    num_batches = int(np.ceil(len(img_list) / batch_size))
    checkpoint_path = r"..\trained\resnet_v2_50.ckpt"
    print("running the model")

    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        arg_scope = resnet_utils.resnet_arg_scope()

        # eval_inputs = tf.random_uniform((batch_size, height, width, 3))
        inputs = tf.placeholder(tf.float32, (None, height, width, 3))

        with slim.arg_scope(arg_scope):
            logits, end_points = resnet_v2_50(inputs, is_training=False)
        predictions = tf.argmax(logits, 1)

        # Create a saver.
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)

            # output = sess.run(predictions)
            # batches, batches_path_list = create_image_batches(img_list, img_path_list, batch_size)
            descriptor_list = []
            for batch_num in range(num_batches):
                print("running batch %d/%d" % (batch_num+1, num_batches))
                cur_batch = img_list[batch_num * batch_size:(batch_num + 1) * batch_size]
                images = np.rollaxis(np.stack(cur_batch, axis=3), 3)
                # descriptor = sess.run(end_points['PreLogitsFlatten'], feed_dict={inputs: images})
                # descriptor = sess.run(end_points['Logits'], feed_dict={inputs: images})
                descriptor = sess.run(logits, feed_dict={inputs: images})

                descriptor_list.append(descriptor)
                # print(descriptor)

    descriptors = np.vstack(descriptor_list)
    return descriptors


def cluster_descriptors(data, num_clusters):
    """

    :param data: the descriptors of the images we want to cluster 
    :return: 
    """

    print("dividing to %d clusters" % num_clusters)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels


def save_by_cluster(images, clusters, images_path, save_path):
    res_folder = os.path.join(save_path, "clustering_results")

    for cluster in np.unique(clusters):
        path = os.path.join(res_folder, "%d" % cluster)
        if not os.path.exists(path):
            os.makedirs(path)

    images_fn = [os.path.split(x)[-1] for x in images_path]

    for i, cluster in enumerate(clusters):
        fn = os.path.join(res_folder, "%d" % cluster, images_fn[i])
        cv2.imwrite(fn, images[i][:, :, ::-1])


if __name__ == "__main__":
    # PARAMS:
    img_size = 224
    images_dir = os.path.join(CURRENT_PATH, "..", "data_set", "Zuriel vila")
    output_dir = os.path.join(CURRENT_PATH, "..", "results", "resnet_Zuriel")

    # Get image list
    img_list, img_path_list = get_data_list(images_dir, img_size)
    num_images = len(img_list)
    num_clusters = int(round(np.sqrt(num_images)))
    descriptors = run_model(img_list)
    clustering_labels = cluster_descriptors(descriptors, num_clusters)
    save_by_cluster(img_list, clustering_labels, img_path_list, output_dir)
    print("")