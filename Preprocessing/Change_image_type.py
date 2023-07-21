import os
import glob
import cv2

def check_and_convert_images(path):
    files = glob.glob(path + '/*/*.jpg')
    for file in files:
        try:
            # Load image
            img = cv2.imread(file)

            # Check if image is loaded properly
            if img is None:
                print(f'Failed to load image at: {file}')
                continue

            if img.size == 0:
                print(f'Invalid image at: {file}')
                continue

            # Convert image to .png and save with the same name
            new_file = os.path.splitext(file)[0] + '.png'
            cv2.imwrite(new_file, img)

            # Check if .png file is created successfully
            if os.path.isfile(new_file):
                # If .png file is created successfully, delete original .jpg file
                os.remove(file)
            else:
                print(f'Failed to convert image to .png: {file}')
        except Exception as e:
            print(f'Error processing file {file}: {str(e)}')

base_path = '/media/hamza/7A11699E3440875D/PycharmProjects/Image_processing_usecases/dataset_parts/train_coin'
check_and_convert_images(base_path)
