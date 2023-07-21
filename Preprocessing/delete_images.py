import os
import glob

base_folder = '/media/hamza/7A11699E3440875D/PycharmProjects/Image_processing_usecases/dataset_parts/train_coin'  # Change this to the path of the directory that contains all your folders

# Loop through each folder
for folder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder)
    if os.path.isdir(folder_path):
        # Loop through each file in the folder
        for file in os.listdir(folder_path):
            # Check if the file is an image with a number greater than 64 in the filename
            try:
                # Extract the number from the filename (assuming format is NUMBER.jpg)
                number = int(os.path.splitext(file)[0])
                if number > 64:
                    file_path = os.path.join(folder_path, file)
                    print(f'Deleting {file_path}')
                    os.remove(file_path)
            except ValueError:
                # If the file name is not a number, skip it
                continue
