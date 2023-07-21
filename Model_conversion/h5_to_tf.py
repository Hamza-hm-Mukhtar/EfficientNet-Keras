import os
import json
import tensorflow as tf

with open('conf.json') as f:
    config = json.load(f)
# Config variables
model_path = config["model_path"]

# Load the trained model
model = tf.keras.models.load_model(os.path.join(model_path, "save_model_stage2.h5"))

# Convert the model to TensorFlow SavedModel format
export_path = os.path.join(model_path, "tf_model")

# Make sure the folder is empty
if os.path.isdir(export_path):
    print(f"\n{export_path} already exists. Please delete it manually to avoid overwriting.")
else:
    model.save(export_path, save_format='tf')
    print(f"[INFO] Model saved in TensorFlow SavedModel format at: {export_path}")
