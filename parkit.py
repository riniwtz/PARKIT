from ultralytics import YOLO
from sort.sort import Sort
from paddleocr import PaddleOCR
from self_utils import *
from logging_cfg import *
import cv2
import time
import numpy as np

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
                print(f"{clean_plate_number(plate_number)}")
    else:
        # Log error if the image is not valid
        log.error("Not a valid image")

# Get license plate from tracked vehicles
def get_license_plate_from_tracked_vehicles(license_plate_detection, tracked_vehicles):
    # Get detected license plate coordinates
    x1, y1, x2, y2, score, class_id = license_plate_detection

    # Assign license plates to vehicles based on coordinates
    for index in range(len(tracked_vehicles)):
        xcar1, ycar1, xcar2, ycar2, car_id = tracked_vehicles[index]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return x1, y1, x2, y2, score
    return 0, 0, 0, 0, 0

# Filter objects to accept ["car", "motorbike", "bus", "truck"] vehicles only
def is_vehicle(class_id):
    # Vehicle IDs: ["car", "motorbike", "bus", "truck"] from coco.names
    vehicle_coco_ids = [2, 3, 5, 7]
    return int(class_id) in vehicle_coco_ids

# Extracts bounding box coordinates of detected vehicles for SORT tracking algorithm
# Returns a list of detected vehicles with their coordinates and scores
def extract_detected_vehicles(object_detections):
    detected_vehicles = []
    for objects in object_detections.boxes.data.tolist():
        # Extract objects data
        x1, y1, x2, y2, score, class_id = objects

        # Perform checks if there is a detected vehicle
        if is_vehicle(class_id):
            # Extract group of detected vehicle coordinates for SORT tracker
            detected_vehicles.append([x1, y1, x2, y2, score])
    return detected_vehicles

# Draws a rectangle on the given video frame with specified coordinates, color, and thickness
def draw_rectangle_to_frame(video_frame, x1, y1, x2, y2, color, thickness):
    cv2.rectangle(video_frame, (x1, y1), (x2, y2), color, thickness)

# Returns a list of bounding box data from object detections
def get_detection_data_box(detections):
    return detections.boxes.data.tolist()

# Updates the SORT vehicle tracker with detected vehicles if the class_id corresponds to a vehicle
def update_vehicle_tracker(class_id, detected_vehicles, vehicle_tracker):
    # Perform checks if there is a detected vehicle first before updating the vehicle tracker to improve system and frames performance
    if is_vehicle(class_id):
        try:
            return vehicle_tracker.update(np.asarray(detected_vehicles))                 
        except Exception as e:
            log.exception("An error occurred with updating object tracker")

# Draws rectangles for each tracked vehicle on the given video frame
def draw_tracked_vehicles(video_frame, tracked_vehicles):
    # Get every tracked vehicle data
    for index in range(len(tracked_vehicles)):
        xcar1, ycar1, xcar2, ycar2, score = tracked_vehicles[index] 
        draw_rectangle_to_frame(video_frame, int(xcar1), int(ycar1)), (int(xcar2), int(ycar2), (0, 0, 255), 2)                         

# Updates the SORT vehicle tracker with detected vehicles and draws the tracked vehicles on the video frame
def update_and_draw_tracked_vehicles(video_frame, object_detections, vehicle_tracker, detected_vehicles):
    # This section is for extracting multiple bounding box coordinates of detected vehicles for SORT tracking algorithm
    # Perform checks if object_detection_data_box is not empty first before updating the vehicle tracker to improve system and frames performance
    object_detections_box = get_detection_data_box(object_detections)
    if object_detections_box:
        # Iterate over detected vehicles
        for objects in object_detections_box:
            # Extract vehicle data
            x1, y1, x2, y2, score, class_id = objects
            tracked_vehicles = update_vehicle_tracker(class_id, detected_vehicles, vehicle_tracker)
            if tracked_vehicles is not None: draw_tracked_vehicles(video_frame, tracked_vehicles)
            return tracked_vehicles

# Crops the video frame based on specified coordinates
def crop_frame(video_frame, x1, y1, x2, y2):
    return video_frame[y1:y2, x1:x2]

# Perform checks if tracked_vehicles is not none before performing license plate detections to improve performance
def perform_license_plate_processing(video_frame, start_time, tracked_vehicles, license_plate_detector):
    if tracked_vehicles is not None:
        # Start license plate detections
        license_plate_detections = license_plate_detector(video_frame, verbose=False, device="mps")[0]

        # Get license plate detections data
        plate_detections_data_box = license_plate_detections.boxes.data.tolist()
        for license_plate_detection in plate_detections_data_box:
            # Get license plates only from tracked vehicles
            x1, y1, x2, y2, score = get_license_plate_from_tracked_vehicles(license_plate_detection, tracked_vehicles)
            
            # Minor y adjustment fix
            y_adjustment = -20
            
            # Get whole number coordinates with adjustments
            c_x1, c_y1, c_x2, c_y2 = int(x1), int(y1) + y_adjustment, int(x2), int(y2) + y_adjustment

            # Draw rectangle
            draw_rectangle_to_frame(video_frame, c_x1, c_y1, c_x2, c_y2, (0, 255, 255), 2)  

            # Crop license plate image from video frame
            license_plate_image = crop_frame(c_y1, c_y2, c_x1, c_x2)

            # Perform OCR processing
            perform_ocr(license_plate_image)

            # Get end process time
            end_time = time.time()

            # Print processing elapsed time
            log.info(f"Processing elapsed time: {end_time - start_time} s")
            
# Start main system processing of vehicle and license plate detection, tracking, and extraction
def start_processing():

    # Initialize real-time trackers for vehicles and license plates
    vehicle_tracker = Sort()

    # Initialize YOLO-based official nano model object detector
    object_detector = YOLO("./models/yolov8n.pt", verbose=False)

    # Initialize YOLO-based pre-trained custom model for license plate detector
    license_plate_detector = YOLO("./models/license_plate_detector.pt", verbose=False)

    # Initialize video capture (can be replaced with a video file or camera index)
    video = cv2.VideoCapture(0)

    # Start real-time processing loop
    while True:
        # Get initial time for performance measurement
        start_time = time.time()

        # Read video frame
        video_status, video_frame = video.read()
             
        # Check if video frame is successfully read
        if not video_status:
            print("Cannot read frame")
        else:
            # Perform vehicle detections using YOLO
            object_detections = object_detector(video_frame, verbose=False, device="mps")[0]

            # Extracts bounding box coordinates of detected vehicles for SORT tracking algorithm
            detected_vehicles = extract_detected_vehicles(object_detections)

            # Update the tracked vehicles with detected vehicles and draw rectangles on the video frame
            tracked_vehicles = update_and_draw_tracked_vehicles(video_frame, object_detections, vehicle_tracker, detected_vehicles)

            # Check if there are tracked vehicles for further processing
            if tracked_vehicles is not None:
                # Perform license plate detections, drawing rectangles, cropping, and OCR processing for tracked vehicles
                perform_license_plate_processing(video_frame, start_time, tracked_vehicles, license_plate_detector)

            # Clear detected vehicles to improve system performance
            detected_vehicles.clear()

            # Update frame
            cv2.imshow('frame', video_frame)
            if cv2.waitKey(1) == ord('q'):
                break

    # Release video resources and destroy all windows to prevent memory leaks
    video.release()
    cv2.destroyAllWindows()

# Starts the main process
start_processing()
