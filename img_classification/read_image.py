import numpy as np
import cv2
import requests
import time

# from src.upload_image_to_s3 import upload_image_to_s3

def read_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for any HTTP error)
        nparr = np.fromstring(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        current_time = int(time.time())
        image_name = url.split('?')[0].split("/")[-1]
        image_name = f"{current_time}_{image_name}"
        # upload_image_to_s3(image_name, image)
        return image
    except requests.exceptions.RequestException as e:
        print("Error fetching the image:", e)
        return None