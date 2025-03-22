import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

def predict(x_data):
    predictions = []
    for x in x_data:
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], np.array([[x]], dtype=np.float32))
        interpreter.invoke()
        # Get the output tensor
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output[0][0])
    return np.array(predictions)

MODELS_DIR = 'models_complex'
MODELS_DIR_RELATIVE = f'../{MODELS_DIR}/'

interpreter = tf.lite.Interpreter(model_path=f"{MODELS_DIR_RELATIVE}/sine_model_quantized.tflite")
interpreter.allocate_tensors()

print('yay i didnt crash')

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()