import tensorflow as tf
from src.tools.dataset import manual_get_datasets
from src.settings.settings import paths
from src.settings.settings import logging, efficientnetb4_params
import os
import numpy as np


def convert_to_fp(model, output_model_name):
    """Converts the model to a TF Lite format without quantization.

    Parameters
    ----------
    model : tf.keras.model
        Keras model.
    output_model_name : str
        Name of the TF Lite model.
    """
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # quantize to float16
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Save the model.
    with open(f'tflite_models/{output_model_name}.tflite', 'wb') as f:
        f.write(tflite_model)


def convert_to_quant(model, output_model_name):
    """Converts the model to a quantized int8 model.

    Parameters
    ----------
    model : tf.keras.model
        Model to quantize.
    output_model_name : str
        Name of the model.
    """

    def representative_data_gen():
        params = efficientnetb4_params
        data_path = paths["data"]["DATA_DIR"]
        train_dir = os.path.join(data_path, params["dataset"], "train_directory")
        test_dir = os.path.join(data_path,  params["dataset"], "test_directory")
        train_dataset, validation_dataset = manual_get_datasets(train_dir, test_dir, params)
        for train_images, _ in train_dataset.take(100):
            # Model has only one input so each data point has one element.
            yield [tf.dtypes.cast(train_images, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.experimental_new_quantizer = True

    tflite_model = converter.convert()

    # Save the model.
    with open(f'tflite_models/{output_model_name}.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':

    # filepath = './best_model.h5'
    # filepath = './mlruns/4/36c49bb524864f17b1fcaa6d0fd15af6/artifacts/model/data/model.h5'
    filepath = './model.h5'

    model = tf.keras.models.load_model(
        filepath, custom_objects=None, compile=True, options=None)
    convert_to_fp(model, 'model')
