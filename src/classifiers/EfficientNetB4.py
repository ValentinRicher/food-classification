"""EfficientNetB4 
https://arxiv.org/pdf/1905.11946v5.pdf
"""


import tensorflow as tf
from src.classifiers.KerasClassifier import KerasClassifier
from src.settings.settings import logging


class EfficientNetB4(KerasClassifier):
    """EfficientNet model used for classify 101 classes of food from food-101 dataset.
    It uses MobileNetV2 architecture with weights trained on ImageNet.
    Only the last layers have been modified to suit the used case.
    """

    def __init__(self, transfer_learning, fine_tuning, input_shape, weights):
        super().__init__()
        if transfer_learning:
            self.model = self.create_model(input_shape)
        elif fine_tuning: # fine-tuning only
            self.model = tf.keras.load_model(weights)
        else:
            raise("Transfer learning or fine tuning or both must be selected")

        # import time
        # print("Load model")
        # time.sleep(30)

    def create_model(
        self,
        input_shape,
    ):
        """Creates the model loaded in the __init__() function.

        Parameters
        ----------
        input_shape :
            Shape of the input image.

        Returns
        -------
        model : keras.model
            The Keras model.
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = self.preprocess_input(inputs)
        base_model = tf.keras.applications.EfficientNetB4(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
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
        x = tf.keras.applications.efficientnet.preprocess_input(x)
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
