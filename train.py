import os
import glob
import json
import time
import datetime
import logging
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint

from utils import generate_batches, create_class_mapping, create_folders

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load user configs
with open('conf.json') as f:
    config = json.load(f)

# Config variables
train_path = config["train_path"]
model_path = config["model_path"]
batch_size = config["batch_size"]
epochs = config["epochs"]
epochs_after_unfreeze = config["epochs_after_unfreeze"]
checkpoint_period = config["checkpoint_period"]
checkpoint_period_after_unfreeze = config["checkpoint_period_after_unfreeze"]

create_folders(model_path)

# Create model
base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
top_layers = base_model.output
top_layers = GlobalAveragePooling2D()(top_layers)
top_layers = Dense(1024, activation='relu')(top_layers)

# Create class mapping
class_mapping = create_class_mapping(train_path)
num_classes = len(class_mapping)
predictions = Dense(num_classes, activation='softmax')(top_layers)

model = Model(inputs=base_model.input, outputs=predictions)
print("[INFO] Successfully loaded base model and model...")

# Create callbacks
checkpoint_path = "logs/weights.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True, period=checkpoint_period)

# Start time
start = time.time()

# Freezing the base layers. Unfreeze the top 2 layers...
for layer in model.layers[:-3]:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Load weights if checkpoint exists
if os.path.exists(checkpoint_path):
    print("[INFO] Resuming training from checkpoint...")
    model.load_weights(checkpoint_path)
else:
    print("[INFO] No previous checkpoint. Starting training from scratch...")

# Start training
print("Start training...")
files = glob.glob(train_path + '/*/*png')
samples = len(files)

# model.fit(generate_batches(train_path, batch_size), epochs=epochs,
#           steps_per_epoch=samples // batch_size, verbose=1, callbacks=[checkpoint])
#
# # Save the model after the first training phase
# print("Saving...")
# model.save(os.path.join(model_path, "save_model_stage1.h5"))

# Unfreeze all layers for fine-tuning if additional epochs are specified
if epochs_after_unfreeze > 0:
    print("Unfreezing all layers...")
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # Load weights if checkpoint exists
    if os.path.exists(checkpoint_path):
        print("[INFO] Resuming training from checkpoint...")
        model.load_weights(checkpoint_path)
    else:
        print("[INFO] No previous checkpoint. Starting training from scratch...")

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss',
                                 save_best_only=True, period=checkpoint_period_after_unfreeze)
    model.fit(generate_batches(train_path, batch_size), epochs=epochs_after_unfreeze,
              steps_per_epoch=samples // batch_size, verbose=1, callbacks=[checkpoint])

    print("Saving...")
    model.save(os.path.join(model_path, "save_model_stage2.h5"))

# end time
end = time.time()
print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
print ("[STATUS] total duration: {}".format(end - start))
