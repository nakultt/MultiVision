import cv2
import time
from ultralytics import YOLO
from modules.email_utils import send_detection_email

def load_specified_objects(filepath="objects.txt"):
    try:
        with open(filepath, "r") as f:
            result = set()
            for line in f:
                if line.strip():
                    result.add(line.strip())
            return result
    except Exception:
        return set()

specified_objects = load_specified_objects()
COOLDOWN = 180  # seconds
last_sent = {}

model = YOLO("yolo12l.pt")

def detect_obj(vid_src=0):
    vid = cv2.VideoCapture(vid_src)
    if not vid.isOpened():
        print(f"Error: Unable to open video source {vid_src}")
        return

    while True:
        success, frame = vid.read()
        if not success:
            print("Error: Unable to read frame")
            break

        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = model.names[class_id]

                if confidence > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Email logic
                    if label in specified_objects:
                        now = time.time()
                        last_time = last_sent.get(label, 0)
                        if now - last_time > COOLDOWN:
                            # Crop the detected object from the frame (corrected slicing)
                            obj_img = frame[y1:y2, x1:x2].copy()
                            send_detection_email(label, confidence, obj_img)
                            last_sent[label] = now

        cv2.imshow("MultiVision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

detect_obj(0)