import tensorflow as tf
import cv2
import os
import sys

from models.resnet.resnet_v2 import resnet_v2_50
from models.resnet import resnet_utils
from sklearn.cluster import KMeans
import numpy as np
<<<<<<< HEAD
import display.album_display as album_display
=======
import src.album_display as disp
import matplotlib.pyplot as plt
>>>>>>> refs/remotes/origin/master

slim = tf.contrib.slim

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)


def get_data_list(dir_path, size):
    """
    :param dir_path: images directory path
    :param size: image target size 
    :return: list of images in a uniform size (of size x size), a list of the original images and a list of their full path
    """

    images = os.listdir(dir_path)
    resized_img_list = []
    img_path_list = []
    # TODO: add ignore from every non fn
    for fn in images:
        print("resizing image %d/%d" % (len(resized_img_list) + 1, len(images)))
        cur_fn = os.path.abspath(os.path.join(dir_path, fn))
        img = cv2.imread(cur_fn, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        resized_img_list.append(pad_resize(img, size))
        img_path_list.append(cur_fn)

    return resized_img_list, img_path_list


def pad_resize(rgb_img, size):
    """
        :param rgb_img: a rgb image
        :param size: the targeted size for resizing
        :return: a square rgb image resize to the desirable size 
    """
    size_x = rgb_img.shape[1]
    size_y = rgb_img.shape[0]
    center = (np.uint32(size_y / 2), np.uint32(size_x / 2))

    r = float(size) / rgb_img.shape[1]

    if size_x < size_y:
<<<<<<< HEAD
        diff = (size_y - size_x) / 2.0
        left = int(np.ceil(diff))
        right = int(np.floor(diff))
        padded = np.lib.pad(rgb_img, ((0, 0), (left, right), (0, 0)), 'constant', constant_values=0)
        # padded = np.lib.pad(rgb_img, ((0, 0), (left, right), (0, 0)),  'median')
=======
        dim = (int(rgb_img.shape[1] * r), size)
        resized = cv2.resize(rgb_img, dim, interpolation=cv2.INTER_AREA)

        diff = (dim[1]-dim[0])/2.0
        left = int(np.ceil(diff))
        right = int(np.floor(diff))
        padded = np.lib.pad(resized, ((0, 0), (left, right), (0, 0)),  'constant', constant_values=0)
>>>>>>> refs/remotes/origin/master

    else:
        dim = (size, int(rgb_img.shape[0] * r))
        resized = cv2.resize(rgb_img, dim, interpolation=cv2.INTER_AREA)

        diff = (dim[0]-dim[1])/2.0
        top = int((np.ceil(diff)))
        buttom = int(np.floor(diff))
<<<<<<< HEAD
        padded = np.lib.pad(rgb_img, ((top, buttom), (0, 0), (0, 0)), 'constant', constant_values=0)
        # padded = np.lib.pad(rgb_img, ((top, buttom), (0, 0), (0, 0)),  'median')

    resized = cv2.resize(padded, (size, size))

    # TODO: add the code below or erase it.
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(resized)
    # plt.show()
=======
        padded = np.lib.pad(resized, ((top, buttom), (0, 0), (0, 0)),  'constant', constant_values=0)
>>>>>>> refs/remotes/origin/master

    return padded


def fit_image(rgb_img, size):
    """
    given an RGB image, return the new image in the given size, 
    using crop from the center and resize

    :param rgb_img: the image that should be resized
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


def create_image_batches(img_list, path_list, batch_size):
    """
    :param img_list: list of images
    :param path_list: the paths to the images in img_list 
    :param batch_size: maximum size of wanted batch
    :return: list of batches of images, where each batch maximum size is batch_size
    """
    batch_num = np.uint32(np.ceil(len(img_list) / batch_size))
    batches = []
    paths_batches = []
    for i in range(batch_num):
        batches.append(img_list[i * batch_size: ((i + 1) * batch_size)])
        paths_batches.append(path_list[i * batch_size: ((i + 1) * batch_size)])
    return batches, paths_batches


def run_model(img_list):
    """
    Gets image batches, run the pretrained resnet_v2 on all the
    images, and return batches of descriptors (one for each image)
    :param img_list: list of images
    #:param batches: image batches 
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
                print("running batch %d/%d" % (batch_num + 1, num_batches))
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


<<<<<<< HEAD
# TODO: check if redundent
# def save_by_cluster(images, clusters, images_path, save_path):
#     res_folder = os.path.join(save_path, "clustering_results")
#     folder_num = 1
#     while os.path.exists(res_folder):
#         res_folder = os.path.join(save_path, "clustering_results_%d" %folder_num)
#         folder_num = folder_num + 1
#
#     for cluster in np.unique(clusters):
#         path = os.path.join(res_folder, "%d" % cluster)
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#     images_fn = [os.path.split(x)[-1] for x in images_path]
#
#     for i, cluster in enumerate(clusters):
#         fn = os.path.join(res_folder, "%d" % cluster, images_fn[i])
#         cv2.imwrite(fn, images[i][:, :, ::-1])

def get_best_descriptor_representations(cluster_descriptors):
    """
        :param cluster_descriptors: descriptors of images in a cluster
        :return: avg_sort: a list of the descriptors sorted in decreased order of compatibility
    """
=======
def get_best_descriptor_representations_list(cluster_descriptors):
>>>>>>> refs/remotes/origin/master
    avg_descriptor = np.mean(cluster_descriptors, axis=0)
    dist_from_avg = np.sum(np.abs(cluster_descriptors - avg_descriptor), axis=1)

    dist_grade = 1/(dist_from_avg + 0.0001)
    best_dist = max(dist_grade)
    dist_grade = dist_grade/best_dist

    return dist_grade


def get_highest_contrast_list(images_paths):
    img_contrasts = []
    for path in images_paths:
        img_gray = cv2.cvtColor(cv2.imread(path[0]), cv2.COLOR_RGB2GRAY)
        avg = np.mean(img_gray)
        var = np.mean(np.square(img_gray - avg))
        img_contrasts.append(np.sum(var))
        # img = cv2.imread(path[0])
        # im_vec = np.vstack(img)
        # avg = np.mean(im_vec, axis=0)
        # im_vec = np.vstack(np.square(img - avg))
        # var = np.mean(im_vec, axis=0)
        # img_contrasts.append(np.sqrt(np.sum(var)))

    best_contrast = max(img_contrasts)
    contrast_array = np.array(img_contrasts)
    contrast_grade = contrast_array / best_contrast

    return contrast_grade

def get_highest_resolution_list(images_paths):
    img_resolutions = []
    for path in images_paths:
        img = cv2.imread(path[0])
        res = img.shape[0] * img.shape[1]
        img_resolutions.append(res)

<<<<<<< HEAD

def get_representing_images_paths(img_list, img_path_list, descriptors, clustering_labels):
=======
    best_resolution = max(img_resolutions)
    resultion_array = np.array(img_resolutions)
    resolution_grade = resultion_array / best_resolution

    return resolution_grade


def get_representing_images_paths(img_path_list, descriptors, clustering_labels):
>>>>>>> refs/remotes/origin/master
    representing_images_path = []
    paths = np.vstack(img_path_list)

    for cluster in np.unique(clustering_labels):
        cluster_paths = paths[clustering_labels == cluster]
        # indices according to the minimal distance from the average descriptor
        desc_dist_grades_list = get_best_descriptor_representations_list(descriptors[clustering_labels == cluster])
        # indices according to the highest contrast of the cluster's images
        contrast_grades_list = get_highest_contrast_list(paths[clustering_labels == cluster])
        # indices according to the highest resolution of the cluster's images
        resolution_grades_list = get_highest_resolution_list(paths[clustering_labels == cluster])

        images_grades = 0.5*desc_dist_grades_list + 0.3*contrast_grades_list + 0.2*resolution_grades_list
        grade_indices = np.argsort(images_grades)[::-1]

        representing_images_path.append(cluster_paths[grade_indices[0]])
    return representing_images_path


def save_representing_images(images_path, save_path):
    res_folder = os.path.join(save_path, "album_results")
    folder_num = 1
    while os.path.exists(res_folder):
        res_folder = os.path.join(save_path, "album_results_%d" % folder_num)
        folder_num = folder_num + 1

    os.makedirs(res_folder)

    for path in images_path:
        fn = os.path.basename(path[0])
        rep_img = cv2.imread(path[0])
        res_path = os.path.join(res_folder, fn)
        cv2.imwrite(res_path, rep_img)

def create_album(album_name, images_dir, output_dir, num_of_selected_images=None):
    img_size = 224

    # Get image list
    resized_img_list, img_path_list = get_data_list(images_dir, img_size)
    num_of_images = len(resized_img_list)

    if num_of_selected_images is None:
        num_of_clusters = int(round(np.sqrt(num_of_images)))
    else:
        num_of_clusters = num_of_selected_images

    descriptors = run_model(resized_img_list)
<<<<<<< HEAD
    clustering_labels = cluster_descriptors(descriptors, num_clusters)
    # save_by_cluster(resized_img_list, clustering_labels, img_path_list, output_dir)
    represetives_path = get_representing_images_paths(original_img_list,
                                                      img_path_list,
=======
    clustering_labels = cluster_descriptors(descriptors, num_of_clusters)

    represetives_path = get_representing_images_paths(img_path_list,
>>>>>>> refs/remotes/origin/master
                                                      descriptors,
                                                      clustering_labels)
    save_representing_images(represetives_path, output_dir)

    represetives_path_list = np.hstack(represetives_path)
    # Create HTML display of the selected images
<<<<<<< HEAD
    album_display.create_album_display(img_path_list, clustering_labels, represetives_path_list,
                                       input_folder, clustering=True)
    print("")
=======
    disp.create_album_display(img_path_list, clustering_labels, represetives_path_list, album_name, clustering=True)

if __name__ == "__main__":
    # PARAMS:
    album_name = "ilan Is 60"
    images_dir = os.path.join(CURRENT_PATH, "..", "data_set", album_name, "all")
    output_dir = os.path.join(CURRENT_PATH, "..", "results", album_name)
    num_of_images = 28

    create_album(album_name, images_dir, output_dir, num_of_images)
>>>>>>> refs/remotes/origin/master
