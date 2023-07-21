import os
import json
import coremltools as ct
import tensorflow as tf

with open('conf.json') as f:
    config = json.load(f)
# Config variables
model_path = config["model_path"]

# Load the trained model
model = tf.keras.models.load_model(os.path.join(model_path, "save_model_stage2.h5"))

# Convert the model to Core ML format
coreml_model = ct.convert(model, source='tensorflow')

# Save the Core ML model
coreml_model.save(os.path.join(model_path, "model.mlmodel"))

print(f"[INFO] Model saved in Core ML format at: {os.path.join(model_path, 'model.mlmodel')}")
