import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils import create_class_mapping
import numpy as np
# Load user configs
with open('conf.json') as f:
    config = json.load(f)

# Config variables
model_path = config["model_path"]
train_path = config["train_path"]

# Create class mapping
class_mapping = create_class_mapping(train_path)
num_classes = len(class_mapping)

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

image_path = "/media/hamza/7A11699E3440875D/PycharmProjects/Image_processing_usecases/dataset_parts/test_coin/2016_Planet_Erde_Rueckseite/3.png"  # Replace with your image path
img_array = preprocess_image(image_path)

# Make predictions
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Map the predicted class index to its label
class_label_mapping = {v: k for k, v in class_mapping.items()}
predicted_class_label = class_label_mapping[predicted_class]

print(f"Predicted class: {predicted_class_label}")
