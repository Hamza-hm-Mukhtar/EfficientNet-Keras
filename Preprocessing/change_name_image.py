import cv2
import os

# Set the directory where the images are stored
image_dir = './Original_images'

# Set the output directory for the images
output_dir = './output_images'

# Set the starting value for the image counter
image_counter = 1

# Iterate over all the images in the directory
for image_filename in os.listdir(image_dir):
    # Read the image using OpenCV
    image = cv2.imread(os.path.join(image_dir, image_filename))

    # Generate a new name for the image
    new_name = f'video30_{image_counter}.jpg'

    # Increment the image counter
    image_counter += 1

    # Save the image to the output directory with the new name
    cv2.imwrite(os.path.join(output_dir, new_name), image)
    print(new_name)
