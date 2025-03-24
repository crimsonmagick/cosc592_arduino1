# Set the root directory for model files
MODEL_DIR="models_complex"
# Save the file as a C source file
xxd -i "$MODEL_DIR/sine_model_quantized.tflite" > "$MODEL_DIR/sine_model_quantized.cc"

# Print the source file
cat "$MODEL_DIR/sine_model_quantized.cc"
