import os
import cv2

path_folder = '/media/hamza/7A11699E3440875D/PycharmProjects/Image_processing_usecases/dataset_parts/'  # Change this to the path of the directory that contains all your folders

# Loop through each folder
for i in range(1, 14):

    base_folder= os.path.join(path_folder, 'coin'+str(i))
    print(base_folder)
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            # Loop through each file in the folder
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # Try to open the image
                try:
                    img = cv2.imread(file_path)
                    # Check if the image was properly loaded
                    if img is None:
                        raise IOError(f"Unable to load image at: {file_path}")

                    # Check if the image has .jpeg or .png extension and save it as .jpg
                    file_extension = os.path.splitext(file)[1].lower()
                    if file_extension in ['.jpeg', '.png']:
                        new_file_path = os.path.join(folder_path, os.path.splitext(file)[0] + '.jpg')
                        cv2.imwrite(new_file_path, img)
                        os.remove(file_path)  # Removing the original file
                        print(f"Saved {file_path} as {new_file_path} and removed the original file.")

                except Exception as e:
                    # If an error occurs in opening the image
                    print(f"Failed to open image: {file_path}. Error: {str(e)}. Removing the file.")
                    os.remove(file_path)  # Removing the file
