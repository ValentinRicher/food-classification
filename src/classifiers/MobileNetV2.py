"""Transfer learning of a MobileNetV2 Keras model.
From https://www.tensorflow.org/tutorials/images/transfer_learning
"""

import base64

import h5py
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from src.classifiers.KerasClassifier import KerasClassifier
from src.settings.settings import logging
from src.tools.utils import decode_and_resize_image
from tensorflow.keras.models import load_model


class MobileNetV2(KerasClassifier):
    """MobileNetV2 model used for classify 101 classes of food from food-101 dataset.
    It uses MobileNetV2 architecture with weights trained on ImageNet.
    Only the last layers have been modified to suit the used case.
    """

    def __init__(self, mobilenetv2_weights, input_shape):
        super().__init__()
        self.model = self.create_model(mobilenetv2_weights, input_shape)
        # import time
        # print("Load model")
        # time.sleep(30)

    def create_model(
        self,
        mobilenetv2_weights,
        input_shape,
    ):
        """Creates the model loaded in the __init__() function.

        Parameters
        ----------
        mobilenetv2_weights : str
            Path to the weights.
        input_shape :
            Shape of the input image.

        Returns
        -------
        model : keras.model
            The Keras model.
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = self.preprocess_input(inputs)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights=mobilenetv2_weights
        )
        base_model.trainable = False
        logging.info("Base model has {} layers".format(len(base_model.layers)))
        x = base_model(x, training=False)
        outputs = self.get_classification_head(x)
        model = tf.keras.Model(inputs, outputs)

        model.summary()

        return model

    def preprocess_input(self, x):
        """Resizes the image.

        Parameters
        ----------
        x : nd.array
            Image

        Returns
        -------
        x : nd.array
            Image
        """
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        # x = tf.keras.layers.experimental.preprocessing.Resizing(
        #     height=160,
        #     width=160,
        #     interpolation="bilinear",
        # )(x)

        return x

    def get_classification_head(self, x):
        """Classification head.

        Parameters
        ----------
        x :
            Tensor from previous layer.

        Returns
        prediction_layer :
            The layer with the probability for each of the 101 classes.
        """
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(global_average_layer)
        prediction_layer = tf.keras.layers.Dense(
            101, activation="softmax", name="prediction_layer"
        )(x)
        return prediction_layer


class InferMobileNetV2(mlflow.pyfunc.PythonModel):
    """Inference model class. This class is made to serve the model with MLFlow
    and must inherit from mlflow.pyfunc.PythonModel.
    """

    def load_context(self, context):
        """When the model is created during the serving, this is the first method to be called,
        like an __init__() method.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            A collection of artifacts that a PythonModel can use when performing inference.
            PythonModelContext objects are created implicitly by the save_model() and log_model()
            persistence methods, using the contents specified by the artifacts parameter of these
            methods.

        Notes
        -----
        From
        https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel.load_context
        Loads artifacts from the specified PythonModelContext that can be used by predict()
        when evaluating inputs. When loading an MLflow model with load_pyfunc(),
        this method is called as soon as the PythonModel is constructed.

        The same PythonModelContext will also be available during calls to predict(),
        but it may be more efficient to override this method and load artifacts
        from the context at model load time.
        """

        with open(context.artifacts["keras_model"], "rb") as file:
            f = h5py.File(file.name, "r")
            self.model = load_model(f)

    def predict(self, context, input):
        """Predict method used by MLFlow during inference.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            Not used here because more efficient to use it in the load_context().
        input : pandas.DataFrame
            DataFrame containing images.
        """

        def decode_img(x):
            if isinstance(x[0], bytes):
                return pd.Series(base64.decodebytes(x[0]))
            else:  # str
                return pd.Series(
                    base64.decodebytes(bytearray(x[0], encoding="utf8"))
                )  # pragma: no cover

        images = input.apply(axis=1, func=decode_img)
        probs = self._predict_images(images)
        return probs

    def _predict_images(self, images):
        """Generate predictions for input images.

        Parameters
        ----------
        images: binary image data

        Returns
        -------
        predicted probabilities for each class
        """

        def preprocess_f(z):
            return decode_and_resize_image(z, (160, 160))

        x = np.array(images[0].apply(preprocess_f).tolist())

        return self.model.predict(x)

