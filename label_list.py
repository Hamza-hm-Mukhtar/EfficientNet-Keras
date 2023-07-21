import os

import json

with open('conf.json') as f:
    config = json.load(f)
# Config variables
train_path = config["train_path"]

# Get class labels
class_labels = sorted(os.listdir(train_path))

# Write class labels to a text file
with open("label.txt", "w") as f:
    for label in class_labels:
        f.write(label + '\n')

print("[INFO] label.txt has been created.")
