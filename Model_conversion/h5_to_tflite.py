import tensorflow as tf
import os
import json

with open('conf.json') as f:
    config = json.load(f)
# Config variables
model_path = config["model_path"]

# Load the trained model
model = tf.keras.models.load_model(os.path.join(model_path, "save_model_stage2.h5"))

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open(os.path.join(model_path, "model.tflite"), 'wb') as f:
    f.write(tflite_model)

print(f"[INFO] Model saved in TFLite format at: {os.path.join(model_path, 'model.tflite')}")
