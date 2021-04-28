from src.tools.dataset import manual_get_datasets
import numpy as np
from src.settings.settings import logging, efficientnetb4_params
from src.settings.settings import paths
import os
import tensorflow as tf 

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(interpreter.get_input_details()[0]['dtype'])
    print(interpreter.get_output_details()[0]['dtype'])

    params = efficientnetb4_params
    data_path = paths["data"]["DATA_DIR"]
    train_dir = os.path.join(data_path, params["dataset"], "train_directory")
    test_dir = os.path.join(data_path,  params["dataset"], "test_directory")
    train_dataset, validation_dataset = manual_get_datasets(train_dir, test_dir, params)

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    test_labels = []
    i = 0
    for test_images, labels in validation_dataset:

        if i % 100 == 0:
            logging.info(f'{i} images have been tested.')

        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.

        test_image = np.expand_dims(test_images[0], axis=0).astype(input_details['dtype'])

        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point
            test_image = test_image.astype(input_details['dtype'])

        interpreter.set_tensor(input_index, test_image)

        import time

        t = time.time()
        # Run inference.
        print('invoke')
        interpreter.invoke()
        print(time.time() - t)

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        print(output)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)
        test_labels.append(np.argmax(labels[0]))

        i += 1

    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_labels[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_digits)

    return accuracy


if __name__ == '__main__':
    # tflite_model_fp32_file = './tflite_models/model.tflite'
    # interpreter_fp32 = tf.lite.Interpreter(model_path=str(tflite_model_fp32_file))
    # interpreter_fp32.allocate_tensors()
    # accuracy_fp32 = evaluate_model(interpreter_fp32)
    # logging.info(f'Accuracy for fp32 model lite {accuracy_fp32}')

    # tflite_model_fp16_file = './tflite_models/model_fp16.tflite'
    # interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))
    # interpreter_fp16.allocate_tensors()
    # accuracy_fp16 = evaluate_model(interpreter_fp16)
    # logging.info(f'Accuracy for fp16 model lite {accuracy_fp16}')

    # tflite_model_quant_file = './tflite_models/model_quant.tflite'
    # interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    # interpreter_quant.allocate_tensors()
    # interpreter_quant = evaluate_model(interpreter_quant)
    # logging.info(f'Accuracy for quant model lite {interpreter_quant}')

    tflite_model_quant_int8_file = './tflite_models/model_quant_int8.tflite'
    interpreter_quant_int8 = tf.lite.Interpreter(model_path=str(tflite_model_quant_int8_file))
    interpreter_quant_int8.allocate_tensors()
    interpreter_quant_int8 = evaluate_model(interpreter_quant_int8)
    logging.info(f'Accuracy for quant model lite {interpreter_quant_int8}')
