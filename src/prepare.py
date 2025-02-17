import tensorflow as tf

# Load the model
loaded_model = tf.keras.models.load_model("../models/model.h5")

# Verify the model structure
loaded_model.summary()