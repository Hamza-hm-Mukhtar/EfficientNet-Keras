import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
with open('conf.json') as f:
    config = json.load(f)
# Config variables
model_path = config["model_path"]

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=os.path.join(model_path, "model.tflite"))
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.ANTIALIAS)
    img_array = np.array(img).astype(np.float32)
    img_array = img_array / 255.0  # Normalise to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Load class labels
with open("label.txt", "r") as f:
    class_labels = f.read().splitlines()

image_path = "/media/hamza/7A11699E3440875D/PycharmProjects/Image_processing_usecases/dataset_parts/test_coin/2016_Planet_Erde_Rueckseite/3.png"  # Replace with your image path
img_array = preprocess_image(image_path)

# Use the model to predict the image class
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data
# Use `tensor()` in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)

# Get the predicted class label
predicted_class_label = class_labels[predicted_class]

print(f"Predicted class: {predicted_class_label}")
