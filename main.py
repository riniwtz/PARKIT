from ultralytics import YOLO
from sort.sort import *
import cv2
from paddleocr import PaddleOCR

def run_ocr(img):
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, det=True, rec=True) # need to run only once to download and load model into memory
    if img is not None and len(img.shape) == 3 and img.shape[0] > 0 and img.shape[1] > 0:
        enhanced_img = cv2.resize(img, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        result = ocr.ocr(enhanced_img, cls=True)
        res = result[0]
        boxes = [line[0] for line in res]
        txts = [line[1][0] for line in res]
        scores = [line[1][1] for line in res]

        for plate_number in txts:
            print(f"Plate Number: {plate_number}, Score: {scores}")
    else:
        print("Error: Invalid image dimensions.")

def get_car(license_plate_detection, tracked_vehicle_ids):
    # Get detected license plate coordinates
    x1, y1, x2, y2, score, class_id = license_plate_detection

    # Assign license plates to cars
    for index in range(len(tracked_vehicle_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = tracked_vehicle_ids[index]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return x1, y1, x2, y2
    return -1, -1, -1, -1
                 
def start_yolo():
    # Vehicle IDs: ["car", "motorbike", "bus", "truck"] from coco.names
    vehicle_coco_ids = [2, 3, 5, 7]

    # Initialize real time tracker
    object_tracker = Sort()

    # Initialize vehicle detector
    vehicle_detector = YOLO("./models/yolov8n.pt", verbose=False)

    # Initialize license plate detector
    license_plate_detector = YOLO("./models/license_plate_detector.pt", verbose=False)

    # Load video from test demo
    video = cv2.VideoCapture("./assets/car_demo.mp4")

    # Load video from webcam
    # webcam = cv2.VideoCapture(0)

    # Check if camera is available
    # if not webcam.isOpened():
    #     print("Cannot open camera")
    # while webcam.isOpened:
    #     pass
  
    while True:
        # Capture frame by frame
        video_status, video_frame = video.read()
        if not video_status:
            print("Cannot read frame")
            break
             
        # Check if webcam can receive frames
        if not video_status:
            print("Cannot read frame")
            break

        # Get detections
        vehicle_detections = vehicle_detector(video_frame, verbose=False)[0]
        detected_vehicles = []
        
        # Iterate over vehicle detections data
        for vehicle_detection in vehicle_detections.boxes.data.tolist():
            # Extract data from array
            x1, y1, x2, y2, score, class_id = vehicle_detection

            # Filter and accept ["car", "motorbike", "bus", "truck"] vehicles only 
            if int(class_id) in vehicle_coco_ids:
                # Include information of the vehicle
                detected_vehicles.append([x1, y1, x2, y2, score])

        # Update real-time object tracker with vehicle detections data
        tracked_vehicle_ids = object_tracker.update(np.asarray(detected_vehicles))

        # Detect license plates
        license_plate_detections = license_plate_detector(video_frame, verbose=False)[0]

        # Iterate over license plate detections
        for license_plate_detection in license_plate_detections.boxes.data.tolist():
            # Crop detected license plate from frame
            crop_x1, crop_y1, crop_x2, crop_y2 = get_car(license_plate_detection, tracked_vehicle_ids)
            license_plate_image = video_frame[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
            run_ocr(license_plate_image)
        
        cv2.imshow('frame', license_plate_image)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

start_yolo()