import tensorflow as tf

print(tf.__version__)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def get_model():
  input = tf.keras.Input(shape=(10,40))

  #No error when dilation rate == 1
  layer = Conv1D(32, (3),dilation_rate =2, padding='same',use_bias=False) (input)
  layer = GlobalMaxPooling1D()(layer)
  output = Dense(2) (layer)

  model = Model(inputs=[input], outputs=[output])
  return model


model = get_model()

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
open("./trained_model.tflite", "wb").write(tflite_model)


interpreter = tf.lite.Interpreter(model_path="./trained_model.tflite")

interpreter.allocate_tensors()


