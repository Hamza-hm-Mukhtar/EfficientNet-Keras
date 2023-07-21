import os
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load user configs
with open('conf.json') as f:
    config = json.load(f)

# Config variables
model_path = config["model_path"]

# Load the trained model
model = tf.keras.models.load_model(os.path.join(model_path, "save_model_stage2.h5"))
print("[INFO] Successfully loaded model...")

# Load and preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Load class labels
with open("label.txt", "r") as f:
    class_labels = f.read().splitlines()

image_path = "/media/hamza/7A11699E3440875D/PycharmProjects/Image_processing_usecases/dataset_parts/test_coin/2016_Planet_Erde_Rueckseite/3.png"  # Replace with your image path
img_array = preprocess_image(image_path)

# Make predictions
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Get the predicted class label
predicted_class_label = class_labels[predicted_class]

print(f"Predicted class: {predicted_class_label}")
