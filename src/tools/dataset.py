import json
import os
import shutil
from pathlib import Path

from src.settings.settings import paths

from PIL import Image
import shutil
import tensorflow as tf
import numpy as np
from pathlib import Path

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

    test_ds = tf.data.Dataset.list_files(str(test_data_dir/"*/*.jpg"), shuffle=True)

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

        # resize the image to the desired size
        img_height = params["img_height"]
        img_width = params["img_width"]
        img = tf.image.resize(img, [img_height, img_width])
        return img

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    # train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    parallel_calls = tf.data.experimental.AUTOTUNE

    # train_ds = train_ds.map(process_path)
    # test_ds = test_ds.map(process_path) 

    train_ds = train_ds.map(process_path, num_parallel_calls=parallel_calls)
    test_ds = test_ds.map(process_path, num_parallel_calls=parallel_calls)

    # train_ds = train_ds.batch(params["batch_size"])
    # test_ds = test_ds.batch(params["batch_size"])

    def configure_for_performance(ds):
        # ds = ds.cache()
        # ds = ds.shuffle(buffer_size=100000)
        ds = ds.batch(params["batch_size"])
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    # train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    # test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

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
