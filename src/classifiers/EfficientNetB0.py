"""Transfer learning of a MobileNetV2 Keras model.
From https://www.tensorflow.org/tutorials/images/transfer_learning
"""


import tensorflow as tf
from src.classifiers.KerasClassifier import KerasClassifier
from src.settings.settings import logging


class EfficientNetB0(KerasClassifier):
    """EfficientNet model used for classify 101 classes of food from food-101 dataset.
    It uses MobileNetV2 architecture with weights trained on ImageNet.
    Only the last layers have been modified to suit the used case.
    """

    def __init__(self, transfer_learning, fine_tuning, input_shape, weights):
        super().__init__()
        if transfer_learning:
            self.model = self.create_model(input_shape)    
        elif fine_tuning: # fine-tuning only
            self.model = tf.keras.models.load_model(weights, compile=False)
        else:
            raise("Transfer learning or fine tuning or both must be selected")

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
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            )
        base_model.trainable = False
        logging.info("Base model has {} layers".format(len(base_model.layers)))
        x = base_model(x, training=False)
        # outputs = self.get_classification_head(x)
        outputs = self.get_new_classification_head(x)
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

    def get_new_classification_head(self, x):
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(x)
        print("avg layer {}".format(global_average_layer.shape))
        global_max_layer = tf.keras.layers.GlobalMaxPooling2D()(x)
        print("max layer {}".format(global_max_layer.shape))

        concatted = tf.keras.layers.Concatenate()([global_average_layer, global_max_layer])
        print("concatted {}".format(concatted.shape))
        # print(concatted)
        # flattened = tf.keras.layers.Flatten()(concatted)
        # print("flattened {}".format(flattened.shape))

        # nn.BatchNorm1d(inp*2,eps=1e-05, momentum=0.1, affine=True)
        bn1 = tf.keras.layers.BatchNormalization()(concatted)
        print("bn1 {}".format(bn1.shape))
        # print(bn1)
        drop1 = tf.keras.layers.Dropout(0.2)(bn1)

        # dense1 = tf.keras.layers.Dense(1280, activation="relu")(drop1)
        # bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(dense1)
        # drop2 = tf.keras.layers.Dropout(0.2)(bn2)

        prediction_layer = tf.keras.layers.Dense(101, activation="softmax", name="prediction_layer")(drop1)

        print(prediction_layer.shape)

        return prediction_layer


if __name__ == "__main__": # to test the classification heads

    input_shape = (8, 7, 7, 1280)
    x = tf.random.normal(input_shape)

    model = EfficientNetB0(True, True, (224, 224, 3), None)

    # pred_layer = model.get_classification_head(x)
    # print(pred_layer.shape)

    new_pred_layer = model.get_new_classification_head(x)
