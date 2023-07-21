import cv2
import os

# Set the directory where the images are stored
image_dir = './Original_images'

# Set the output directory for the images
output_dir = './output_images'

# Check if the output directory exists. If not, create it.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over all the images in the directory
for image_filename in os.listdir(image_dir):
    # Read the image using OpenCV
    image = cv2.imread(os.path.join(image_dir, image_filename))

    # Generate a new name for the image: remove the spaces from the filename
    new_name = image_filename.replace(' ', '')

    # Save the image to the output directory with the new name
    cv2.imwrite(os.path.join(output_dir, new_name), image)
    print(new_name)

