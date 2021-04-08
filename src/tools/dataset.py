import json
import os
import shutil
from pathlib import Path

import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from src.settings.settings import logging, paths
from src.tools import autoaugment
from tensorflow.keras.preprocessing import image_dataset_from_directory


def create_train_test_folders(dest_dir: str, image_file: str) -> None:
    """Creates the dest_dir folder from the list of images in image_file.

    Parameters
    ----------
    dest_dir : str
        The destination folder where to send the images belonging to image_file.
    image_file : str
        The JSON file containing the images.
    """

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.mkdir(dest_dir)

    with open(image_file) as f:
        test = json.load(f)

    for cat in test.keys():
        if not os.path.exists(Path(dest_dir, cat)):
            os.mkdir(Path(dest_dir, cat))

        # parsing of the image name to get only the number and not the category
        test_images = [test_image.split(cat + "/")[1] for test_image in test[cat]]
        for test_image in test_images:

            shutil.copy(
                Path(paths["FOOD_DIR"], "images", cat, test_image + ".jpg"),
                Path(dest_dir, cat, test_image + ".jpg"),
            )

def manual_get_datasets(train_data_dir, test_data_dir, params):
    # https://www.tensorflow.org/tutorials/load_data/images

    train_data_dir = Path(train_data_dir)
    test_data_dir = Path(test_data_dir)

    train_ds = tf.data.Dataset.list_files(str(train_data_dir/"*/*.jpg"), shuffle=True)
    if params["augment"]:
        temp_ds = train_ds
        for _ in range(params["aug_factor"]-1):
            temp_ds = temp_ds.concatenate(train_ds)
        train_ds = temp_ds
    logging.debug("Size of the training dataset: {}".format(len(train_ds)))

    test_ds = tf.data.Dataset.list_files(str(test_data_dir/"*/*.jpg"), shuffle=True)
    logging.debug("Size of the testing dataset: {}".format(len(test_ds)))

    class_names = np.array(sorted([item.name for item in train_data_dir.glob('*') if item.name not in ["LICENSE.txt", ".DS_Store"]]))

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        return one_hot

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    def resize_img(img):
        # resize the image to the desired size
        img_height = params["img_height"]
        img_width = params["img_width"]
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.cast(img, dtype=tf.uint8)
        return img

    def img_aug(img):
        # seq = iaa.Sequential([
        #     iaa.CropAndPad(percent=(-0.25, 0.25)),
        #     iaa.Rotate((-45, 45)),
        #     iaa.Fliplr(0.5),
        # ])
        # # TODO 
        # # add translation
        # # gaussian noise
        # img = seq(image=img)

        img = autoaugment.distort_image_with_autoaugment(img, 'v0')

        return img


    @tf.function(input_signature=[tf.TensorSpec((params["img_height"], params["img_width"], params["img_n_channels"]), tf.uint8)])
    def tf_img_aug(input):
        aug_img = tf.numpy_function(img_aug, [input], tf.uint8)
        return aug_img

    def process_path(file_path, resize, augment):
        """Processes the datasets. 

        Parameters
        ----------
        file_path : str
            Path to the image.
        augment : bool
            If training images needs to be augmented or not.

        Returns
        -------
        img, label
            The image and the label.
        """

        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)

        if resize:
            img = resize_img(img)

        if augment:
            # img = img_aug(img)
            img = tf_img_aug(img)
            img = tf.reshape(img, shape=(params["img_height"], params["img_width"], params["img_n_channels"]))

        return img, label

    parallel_calls = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.map(lambda x: process_path(x, resize=params["resize"], augment=params["augment"]), num_parallel_calls=parallel_calls)
    test_ds = test_ds.map(lambda x: process_path(x, resize=params["resize"], augment=False), num_parallel_calls=parallel_calls)

    def configure_for_performance(ds):
        # ds = ds.cache() # only if the dataset fit in memory
        ds = ds.batch(params["batch_size"])
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds)
    test_ds = configure_for_performance(test_ds)

    return train_ds, test_ds

def auto_get_datasets(train_dir, test_dir, params):

    train_dataset = image_dataset_from_directory(
        train_dir,
        shuffle=True,
        batch_size=params["batch_size"],
        image_size=(
            params["img_height"],
            params["img_width"],
        ),
        label_mode="categorical",
        # interpolation='gaussian'
    )
    validation_dataset = image_dataset_from_directory(
        test_dir,
        shuffle=True,
        batch_size=params["batch_size"],
        image_size=(
            params["img_height"],
            params["img_width"],
        ),
        label_mode="categorical",
        # interpolation='gaussian'
    )

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, validation_dataset
