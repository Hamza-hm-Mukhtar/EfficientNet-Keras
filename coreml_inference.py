import os
import json
import coremltools as ct
import numpy as np
from PIL import Image

with open('conf.json') as f:
    config = json.load(f)
# Config variables
model_path = config["model_path"]

# Load the Core ML model
coreml_model = ct.models.MLModel(os.path.join(model_path, "model.mlmodel"))
print("[INFO] Successfully loaded Core ML model...")

# Load and preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.ANTIALIAS)
    img_array = np.array(img).astype(np.float32)
    img_array = np.transpose(img_array, (2, 0, 1))  # Change to CHW format
    img_array = img_array / 255.0  # Normalise to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Load class labels
with open("label.txt", "r") as f:
    class_labels = f.read().splitlines()

image_path = "/media/hamza/7A11699E3440875D/PycharmProjects/Image_processing_usecases/dataset_parts/test_coin/2016_Planet_Erde_Rueckseite/3.png"  # Replace with your image path
img_array = preprocess_image(image_path)

# Make predictions
predictions = coreml_model.predict({'input_1': img_array})
predicted_class = np.argmax(predictions['Identity'])

# Get the predicted class label
predicted_class_label = class_labels[predicted_class]

print(f"Predicted class: {predicted_class_label}")
