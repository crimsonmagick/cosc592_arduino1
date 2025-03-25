import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
import os

SAMPLES = 1000

x_values = np.random.uniform(low=0, high=2 * math.pi, size=SAMPLES).astype(
  np.float32)

np.random.shuffle(x_values)

y_values = np.sin(x_values).astype(np.float32)

plt.plot(x_values, y_values, 'b.')
plt.show()

y_values += 0.1 * np.random.randn(*y_values.shape)

plt.plot(x_values, y_values, 'b.')
plt.show()

TRAIN_SPLIT =  int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_test, y_test, 'r.', label="Test")
plt.plot(x_validate, y_validate, 'y.', label="Validate")
plt.legend()
plt.show()

model = tf.keras.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(1,)))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss="mse", metrics=["mae"])

history = model.fit(x_train, y_train, epochs=500, batch_size=64,
                        validation_data=(x_validate, y_validate))

train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictions = model.predict(x_train)
plt.clf()
plt.title('Training data predicted vs actual values')
plt.plot(x_test, y_test, 'b.', label='Actual')
plt.plot(x_train, predictions, 'r.', label='Predicted')
plt.legend()
plt.show()


MODELS_DIR = 'models_complex'
MODELS_DIR_RELATIVE = f'../{MODELS_DIR}/'
if not os.path.exists(MODELS_DIR_RELATIVE):
  os.mkdir(MODELS_DIR_RELATIVE)
MODEL_TF = MODELS_DIR_RELATIVE + 'model.h5'
MODEL_NO_QUANT_TFLITE = MODELS_DIR_RELATIVE + 'model_no_quant.tflite'
MODEL_TFLITE = MODELS_DIR_RELATIVE + 'model.tflite'
MODEL_TFLITE_MICRO = MODELS_DIR_RELATIVE + 'model.cc'

model.save(MODEL_TF)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the model to disk
open(f"../{MODELS_DIR}/sine_model.tflite", "wb").write(tflite_model)
# Convert the model to the TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Indicate that we want to perform the default optimizations,
# which include quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Define a generator function that provides our test data's x values
# as a representative dataset, and tell the converter to use it
def representative_dataset_generator():
  for value in x_test:
    yield [np.array(value, dtype=np.float32, ndmin=2)]
converter.representative_dataset = representative_dataset_generator

# Convert the model
tflite_model = converter.convert()
# Save the model to disk
open(f"../{MODELS_DIR}/sine_model_quantized.tflite", "wb").write(tflite_model)

basic_model_size = os.path.getsize(f"../{MODELS_DIR}/sine_model.tflite")
print("Basic model is %d bytes" % basic_model_size)
quantized_model_size = os.path.getsize(f"../{MODELS_DIR}/sine_model_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)
difference = basic_model_size - quantized_model_size
print("Difference is %d bytes" % difference)

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

interpreter = tf.lite.Interpreter(model_path=f"{MODELS_DIR_RELATIVE}/sine_model_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

nonquantized_predictions = model.predict(x_test)
quantized_predictions = predict(x_test)
plt.figure(figsize=(10, 6))
plt.title('Quantized Model Predictions vs Actual Values')
plt.plot(x_test, y_test, 'b.', label='Actual')
plt.plot(x_test, nonquantized_predictions, 'g.', label='Model Prediction')
plt.plot(x_test, quantized_predictions, 'r.', label='Quantized Model Prediction')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()