import base64
import copy
import unittest

import mlflow
import numpy as np
import pandas as pd
from src.classifiers.MobileNetV2 import InferMobileNetV2, MobileNetV2
from src.tools.utils import read_image
from tensorflow.keras.preprocessing import image_dataset_from_directory


class TestMobileNetV2(unittest.TestCase):
    def setUp(self):
        self.mobilenet = MobileNetV2(
            mobilenetv2_weights="/app/list/nasia/Valentin/weights/\
                mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
            input_shape=(160, 160, 3),
        )
        self.train_dataset = image_dataset_from_directory(
            "/app/list/nasia/Valentin/datasets/food-101/train_directory",
            shuffle=True,
            batch_size=8,
            image_size=(
                160,
                160,
            ),
            label_mode="categorical",
        ).take(10)

        self.validation_dataset = image_dataset_from_directory(
            "/app/list/nasia/Valentin/datasets/food-101/test_directory",
            shuffle=True,
            batch_size=8,
            image_size=(
                160,
                160,
            ),
            label_mode="categorical",
        ).take(2)

    def test_train(self):

        non_trained_last_layer = self.mobilenet.model.get_layer("prediction_layer")
        non_trained_weights = copy.deepcopy(non_trained_last_layer.get_weights())

        self.mobilenet.train(self.train_dataset, self.validation_dataset, 2, 0.0001)
        trained_last_layer = self.mobilenet.model.get_layer("prediction_layer")
        trained_weights = copy.deepcopy(trained_last_layer.get_weights())

        self.assertFalse(np.array_equal(non_trained_weights[0], trained_weights[0]))
        self.assertFalse(np.array_equal(non_trained_weights[1], trained_weights[1]))


class TestInferMobileNetV2(unittest.TestCase):
    def setUp(self):

        img_path = "/app/list/nasia/Valentin/datasets/food-101/test_directory/apple_pie/1011328.jpg"
        img_path_2 = "/app/list/nasia/Valentin/datasets/food-101/test_directory/apple_pie/\
            101251.jpg"
        filenames = [img_path, img_path_2]
        self.data = pd.DataFrame(
            data=[base64.encodebytes(read_image(x)) for x in filenames],
            columns=["image"],
        )

        artifacts = {
            "keras_model": "/app/list/nasia/Valentin/richevn/example-image/mlruns/1/\
                00c8877860fd47daa870f2ebae9cb073/artifacts/py_model/artifacts/model.h5",
        }
        self.context = mlflow.pyfunc.PythonModelContext(artifacts)

        self.infermodel = InferMobileNetV2()
        self.infermodel.load_context(self.context)

    def test_predict(self):

        predictions = self.infermodel.predict(self.context, self.data)
        # Check that the number of predictions is the same as the number of input data
        self.assertEqual(len(self.data), len(predictions))
        # Check that the number of classes in predictions is equal to 101
        self.assertEqual(len(predictions[0]), 101)
