import tensorflow as tf
import os

print(tf.__version__)

input_arrays = ['input']
output_arrays = ['output/BiasAdd']

input_shapes = {
  'input': [2, 3]
}

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('freeze_linear_model_output', input_arrays, output_arrays, input_shapes=input_shapes)
converter.allow_custom_ops = True
tflite_model = converter.convert()
open('tf_lite_linear_output', 'wb').write(tflite_model)


