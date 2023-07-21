
import glob
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import os

def create_class_mapping(path):
    # Find all unique class names
    class_names = set(os.path.basename(os.path.dirname(f)) for f in glob.glob(path + '/*/*png'))
    # Create a mapping from class names to integers
    return {class_name: idx for idx, class_name in enumerate(class_names)}

def generate_batches(path, batchSize):
    # Create class mapping
    class_mapping = create_class_mapping(path)
    
    while True:
        files = glob.glob(path + '/*/*png')
        for f in range(0, len(files), batchSize):
            x = []
            y = []
            for i in range(f, f + batchSize):
                if i < len(files):
                    # Load image
                    img = cv2.imread(files[i])
                    
                    # Check if image is loaded properly
                    if img is None or img.size == 0:
                        print(f'Invalid image or failed to load image at: {files[i]}')
                        continue
                    
                    # Resize image
                    x.append(cv2.resize(img, (224, 224)))
                    
                    # Extract class name and convert to integer using mapping
                    class_name = os.path.basename(os.path.dirname(files[i]))
                    class_idx = class_mapping[class_name]
                    y.append(class_idx)
            
            # Check if the batch is not empty
            if len(x) > 0 and len(y) > 0:            
                yield (np.array(x), to_categorical(y, num_classes=len(class_mapping)))

def generate_batches_with_augmentation(train_path, batch_size, validation_split, augmented_data):
        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            rotation_range=0.3,
            zoom_range=0.1,
            validation_split=validation_split)
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(224, 224),
            batch_size=batch_size, 
            save_to_dir=augmented_data)
        return train_generator

def create_folders(model_path):
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if not os.path.exists("logs"):
        os.mkdir("logs")
        