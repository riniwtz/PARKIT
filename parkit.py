from ultralytics import YOLO
from paddleocr import PaddleOCR
from self_utils import *
from logging_cfg import *
import cv2
import time

# Perform OCR on license plate image
def perform_ocr(img):
    # Initialize PaddleOCR for text recognition
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    
    # Check if the input image is valid
    if is_valid_image(img):
        # Perform OCR on the enhanced image using PaddleOCR
        results = ocr.ocr(enhance_image(img))
   
        # Check if OCR results are available
        data = results[0]
        if data is not None:
            # Iterate through OCR results
            for coordinate_box, plate_data in data:
                top_left = coordinate_box[0]
                top_right = coordinate_box[1]
                bottom_right = coordinate_box[2]
                bottom_left = coordinate_box[3]
                plate_number = plate_data[0]
                accuracy = plate_data[1]

                # Print cleaned plate number
                print(f"Plate Number: {clean_plate_number(plate_number)}, Accuracy: {accuracy}")
    else:
        # Log error if the image is not valid
        log.error("Not a valid image")
        
# Filter objects to accept ["car", "motorbike", "bus", "truck"] vehicles only
def is_vehicle(class_id):
    # Vehicle IDs: ["car", "motorbike", "bus", "truck"] from coco.names
    vehicle_coco_ids = [2, 3, 5, 7]
    return int(class_id) in vehicle_coco_ids

# Draws a rectangle on the given video frame with specified coordinates, color, and thickness
def draw_rectangle_to_frame(video_frame, x1, y1, x2, y2, color, thickness):
    cv2.rectangle(video_frame, (x1, y1), (x2, y2), color, thickness)

# Returns a list of bounding box data from object detections
def get_detection_data_box(detections):
    return detections.boxes.data.tolist()

# Crops the video frame based on specified coordinates
def crop_frame(video_frame, x1, y1, x2, y2):
    return video_frame[y1:y2, x1:x2]
            
# Start main system processing of vehicle and license plate detection, tracking, and extraction
def start_processing():

    # Initialize YOLO-based official nano model object detector
    object_detector = YOLO("./models/yolov8n.pt", verbose=False)

    # Initialize YOLO-based pre-trained custom model for license plate detector
    license_plate_detector = YOLO(model="./models/license_plate_detector.pt", verbose=False)

    # Initialize video capture (can be replaced with a video file or camera index)
    video = cv2.VideoCapture(0)
    # video = cv2.imread("/Users/riniwtz/Downloads/jt_car.png")


    # Start real-time processing loop
    while True:
        # Get initial time for performance measurement
        start_time = time.time()

        # Read video frame
        video_status, video_frame = video.read()
        # video_frame = video
        video_status = True
             
        # Check if video frame is successfully read
        if not video_status:
            print("Cannot read frame")
        else:
            # This line calls the object detector function, passing the video frame as input.
            # The `verbose` parameter is set to `False` to suppress output, and the `device` parameter
            # is set to "mps" to use the Metal Performance Shaders (MPS) for hardware acceleration (if available).
            # The result is stored in `object_detections`, which is a list of detected objects.
            object_detections = object_detector(video_frame, verbose=False, device="mps")[0]

            # This line extracts the bounding box data from the first detected object (assuming there's at least one).
            # If no objects are detected, `get_detection_data_box` will return None.
            object_detection_box = get_detection_data_box(object_detections)

            # This conditional statement checks if an object was detected.
            if object_detection_box:
                # Loop through each detected object in the bounding box.
                for car_detection in object_detection_box:
                    # Unpack the bounding box coordinates, confidence score, and class ID from the detection data.
                    c_x1, c_y1, c_x2, c_y2, score, class_id = car_detection

                    # Convert the bounding box coordinates to integers for drawing purposes.
                    cc_x1, cc_y1, cc_x2, cc_y2 = int(c_x1), int(c_y1), int(c_x2), int(c_y2)

                    # Check if the detected class is a vehicle.
                    if is_vehicle(class_id):
                        # Draw a blue rectangle around the detected vehicle on the video frame.
                        draw_rectangle_to_frame(video_frame, cc_x1, cc_y1, cc_x2, cc_y2, (0, 0, 255), 2)

                        # This line calls the license plate detector function, similar to the object detector.
                        license_plate_detections = license_plate_detector(video_frame, verbose=False, device="mps")[0]
                        license_plate_detection_box = get_detection_data_box(license_plate_detections)

                        # Check if a license plate was detected within the vehicle bounding box.
                        if license_plate_detection_box:
                            # Loop through each detected license plate in the bounding box.
                            for license_plate_detection in license_plate_detection_box:
                                # Unpack the bounding box coordinates, confidence score, and class ID from the detection data.
                                p_x1, p_y1, p_x2, p_y2, score, class_id = license_plate_detection

                                # Convert the bounding box coordinates to integers for drawing purposes.
                                cp_x1, cp_y1, cp_x2, cp_y2 = int(p_x1), int(p_y1), int(p_x2), int(p_y2)

                                # Adjust the license plate bounding box to exclude the top 20 pixels (optional, might need adjustment).
                                cp_y1, cp_y2 = cp_y1 - 20, cp_y2 - 20

                                # Check if the license plate bounding box is within the vehicle bounding box.
                                if cp_x1 >= c_x1 and cp_y1 >= c_y1 and cp_x2 <= cc_x2 and cp_y2 <= cc_y2:
                                    # Draw a yellow rectangle around the detected license plate on the video frame.
                                    draw_rectangle_to_frame(video_frame, cp_x1, cp_y1, cp_x2, cp_y2, (0, 255, 255), 2)
                                    video_frame = adjust_gamma(video_frame)
                                    # video_frame = sharpen_img(video_frame)

                                    # Extract the license plate region from the video frame and perform OCR (Optical Character Recognition) on it.
                                    perform_ocr(crop_frame(video_frame, cp_x1, cp_y1, cp_x2, cp_y2))

                                    # Get the current time after the processing is complete.
                                    end_time = time.time()

                                    # Calculate the total elapsed time by subtracting the start time from the end time.
                                    elapsed_time = end_time - start_time  # Calculate the difference in seconds

                                    # Print the elapsed processing time to the console, formatted for readability.
                                    print(f"Elapsed processing time: {elapsed_time:.4f} s")  # Format to 4 decimal places

            # Update frame
            cv2.imshow('frame', video_frame)
            if cv2.waitKey(1) == ord('q'):
                break

    # Release video resources and destroy all windows to prevent memory leaks
    video.release()
    cv2.destroyAllWindows()

# Starts the main process
start_processing()
