from PIL import Image
import numpy as np
import os

current_directory = os.path.join(os.path.dirname(__file__))
path = os.path.join(current_directory, "living_test.jpeg")

def split_panorama(image_path, degrees_increment=5):
    output_dir = os.path.join(current_directory, "split_images")
    # Open the panoramic image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine the number of segments based on the degrees increment
    segments = int(360 / degrees_increment)

    # Calculate the width of each segment
    segment_width = int(image_array.shape[1] / segments)

    # Split the panoramic image into segments and save them
    for i in range(segments):
        start_col = i * segment_width
        end_col = (i + 1) * segment_width
        segment_image_array = image_array[:, start_col:end_col, :]
        segment_image = Image.fromarray(segment_image_array)
        segment_image.save(os.path.join(output_dir, f"segment_{i}.jpg"))  # Save the segment as a separate image

split_panorama(path, 30)
