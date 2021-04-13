import tensorflow as tf

filepath = './mlruns/4/36c49bb524864f17b1fcaa6d0fd15af6/artifacts/model/data/model.h5'
model = tf.keras.models.load_model(
    filepath, custom_objects=None, compile=True, options=None
)
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
