
# Set the root directory for model files
#MODEL_DIR="models"
MODEL_DIR="models_simple"
#MODEL_DIR="models_ib"

# Install xxd if it is not available
# apt-get -qq install xxd

# Save the file as a C source file
xxd -i "$MODEL_DIR/sine_model_quantized.tflite" > "$MODEL_DIR/sine_model_quantized.cc"

# Print the source file
cat "$MODEL_DIR/sine_model_quantized.cc"
