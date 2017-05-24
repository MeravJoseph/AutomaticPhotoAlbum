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
    resized_img_list = []
    original_img_list = []
    img_path_list = []
    for fn in images:
        print("resizing image %d/%d" % (len(resized_img_list)+1, len(images)))
        cur_fn = os.path.abspath(os.path.join(dir_path, fn))
        img = cv2.imread(cur_fn, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 3:
            original_img_list.append(img)
            resized_img_list.append(pad_resize(img, size))
            img_path_list.append(cur_fn)
    return resized_img_list, original_img_list, img_path_list

def pad_resize(rgb_img, size):
    size_x = rgb_img.shape[1]
    size_y = rgb_img.shape[0]
    center = (np.uint32(size_y / 2), np.uint32(size_x / 2))

    if size_x < size_y:
        diff = (size_y-size_x)/2.0
        left = int(np.ceil(diff))
        right = int(np.floor(diff))
        padded = np.lib.pad(rgb_img, ((0, 0), (left, right), (0, 0)),  'constant', constant_values=0)
        # padded = np.lib.pad(rgb_img, ((0, 0), (left, right), (0, 0)),  'median')

    else:
        diff = (size_x - size_y) / 2.0
        top = int((np.ceil(diff)))
        buttom = int(np.floor(diff))
        padded = np.lib.pad(rgb_img, ((top, buttom), (0, 0), (0, 0)),  'constant', constant_values=0)
        # padded = np.lib.pad(rgb_img, ((top, buttom), (0, 0), (0, 0)),  'median')

    resized = cv2.resize(padded, (size, size))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(resized)
    # plt.show()

    return resized

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
    batch_size = 5
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
    folder_num = 1
    while os.path.exists(res_folder):
        res_folder = os.path.join(save_path, "clustering_results_%d" %folder_num)
        folder_num = folder_num + 1

    for cluster in np.unique(clusters):
        path = os.path.join(res_folder, "%d" % cluster)
        if not os.path.exists(path):
            os.makedirs(path)

    images_fn = [os.path.split(x)[-1] for x in images_path]

    for i, cluster in enumerate(clusters):
        fn = os.path.join(res_folder, "%d" % cluster, images_fn[i])
        cv2.imwrite(fn, images[i][:, :, ::-1])

def get_best_descriptor_representations(cluster_descriptors):
    avg_descriptor = np.mean(cluster_descriptors, axis=0)
    dist_from_avg = np.sum(np.abs(cluster_descriptors - avg_descriptor), axis=1)
    avg_sort = np.argsort(dist_from_avg)

    return avg_sort

def get_representing_images_paths(img_list, img_path_list, descriptors, clustering_labels):
    representing_images_path = []
    paths = np.vstack(img_path_list)

    for cluster in np.unique(clustering_labels):
        cluster_paths = paths[clustering_labels == cluster]
        # indices according to the minimal distance from the average descriptor
        desc_min_dist = get_best_descriptor_representations(descriptors[clustering_labels == cluster])
        # TODO: fix image resolution, image contrast etc. and combine with the avg_desc_dist
        # resolustion = []
        # for p in cluster_paths:
        #     img = cv2.imread(p)
        #     resolustion.append(img.shape[0]*img.shape[1])
        #
        # median_img_res = np.median(resolustion)



        representing_images_path.append(cluster_paths[desc_min_dist[0]])
    return representing_images_path


def save_representing_images(images_path, save_path):
    res_folder = os.path.join(save_path, "album_results")
    folder_num = 1
    while os.path.exists(res_folder):
        res_folder = os.path.join(save_path, "album_results_%d" %folder_num)
        folder_num = folder_num + 1

    os.makedirs(res_folder)

    for path in images_path:
        fn = os.path.basename(path[0])
        rep_img = cv2.imread(path[0])
        res_path = os.path.join(res_folder, fn)
        cv2.imwrite(res_path, rep_img)


if __name__ == "__main__":
    # PARAMS:
    img_size = 224
    input_folder = "new_zealand_trip"
    # images_dir = os.path.join(CURRENT_PATH, "..", "data_set", "Zuriel vila")
    images_dir = os.path.join(CURRENT_PATH, "..", "data_set", input_folder)
    output_dir = os.path.join(CURRENT_PATH, "..", "results", input_folder)

    # Get image list
    resized_img_list, original_img_list, img_path_list = get_data_list(images_dir, img_size)
    num_images = len(resized_img_list)
    num_clusters = int(round(np.sqrt(num_images)))
    descriptors = run_model(resized_img_list)
    clustering_labels = cluster_descriptors(descriptors, num_clusters)
    save_by_cluster(resized_img_list, clustering_labels, img_path_list, output_dir)
    avg_represetives_path = get_representing_images_paths(original_img_list,
                                                          img_path_list,
                                                          descriptors,
                                                          clustering_labels)
    save_representing_images(avg_represetives_path, output_dir)
    print("")