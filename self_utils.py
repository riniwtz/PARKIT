
import os
import cv2
from logging_cfg import *

# Checks if the provided image is valid for processing.
def is_valid_image(img):
    return img is not None and img.shape[0] > 0 and img.shape[1] > 0

# Saves the provided image to a file with the given filename in the 'output' directory.
def save_image(filename, img, log):
    output_path = os.path.join("output", filename)
    status = cv2.imwrite(output_path, img)

    if log:
        status_msg = "Saved" if status else "Failed to save"
        log.info("%s to %s" % (status_msg, output_path))

# Cleans the plate number text by converting it to uppercase and removing spaces.
def clean_plate_number(plate_number):
    return "".join(plate_number.upper().split()).strip()

# Enhances the provided image by resizing it twice the size using cubic interpolation.
def enhance_image(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
