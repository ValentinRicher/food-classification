"""To test the use of GPUs
"""
import sys

print("PATHS")
print(sys.path)

import tensorflow as tf
import numpy as np
from src.classifiers.MobileNetV2 import MobileNetV2, InferMobileNetV2
from src.settings.settings import logging, mobilenetv2_params
from tensorflow.keras.preprocessing import image_dataset_from_directory
import argparse
import os


# os.environ['PYTHONPATH'] = ""


def train(mounted_path):

    # from azureml.core import Dataset, Run

    # run = Run.get_context()
    # workspace = run.experiment.workspace

    # Get a dataset by name
    # test_dataset = Dataset.get_by_name(workspace=workspace, name="food-101")
    # print(test_dataset.to_path())
    # test_dataset.download(target_path="/tmp/dataset/")

    # print(mounted_path)
    # print(glob.glob(os.path.join(mounted_path, '**/test_directory/**'), recursive=True))

    mobilenetv2_params["train_dir"] = os.path.join("/tmp/dataset/", "train_directory")
    print(mobilenetv2_params["train_dir"])
    mobilenetv2_params["test_dir"] = os.path.join("/tmp/dataset/", "test_directory")
    print(mobilenetv2_params["test_dir"])
    mobilenetv2_params["mobilenetv2_weights"] = None

    # print(os.listdir(mounted_path))

    mobilenet = MobileNetV2(
        mobilenetv2_params["mobilenetv2_weights"],
        (
            mobilenetv2_params["img_height"],
            mobilenetv2_params["img_width"],
            mobilenetv2_params["img_n_channels"],
        ),
    )

    train_dataset = image_dataset_from_directory(
        mobilenetv2_params["train_dir"],
        shuffle=True,
        batch_size=mobilenetv2_params["batch_size"],
        image_size=(
            mobilenetv2_params["img_height"],
            mobilenetv2_params["img_width"],
        ),
        label_mode="categorical",
    )

    validation_dataset = image_dataset_from_directory(
        mobilenetv2_params["test_dir"],
        shuffle=True,
        batch_size=mobilenetv2_params["batch_size"],
        image_size=(
            mobilenetv2_params["img_height"],
            mobilenetv2_params["img_width"],
        ),
        label_mode="categorical",
    )

    # if mobilenetv2_params["test_mode"]:
    #     train_dataset = train_dataset.take(100)
    #     validation_dataset = validation_dataset.take(20)

    history = mobilenet.train(
        train_dataset,
        validation_dataset,
        n_epochs=mobilenetv2_params["n_epochs"],
        learning_rate=mobilenetv2_params["learning_rate"],
    )


if __name__ == "__main__":
    # retrieve the 2 arguments configured through `arguments` in the ScriptRunConfig
    parser = argparse.ArgumentParser()
    parser.add_argument("--mounted-path", type=str, dest="mounted_path")
    # parser.add_argument('--test-dir', type=str, dest='test_dir')
    args = parser.parse_args()
    mounted_path = args.mounted_path
    # test_dir = args.test_dir

    train(mounted_path)
