import cv2
from object_detection.vision import detect_obj_on_frame
from face_detection.train_faces import train_model
from face_detection.detect_faces import recognize_faces_on_frame, load_model

if __name__ == "__main__":
    known_encodings, known_names = load_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Failed to capture frame from webcam. Retrying...")
            continue
        frame = detect_obj_on_frame(frame)
        frame = recognize_faces_on_frame(frame, known_encodings, known_names)
        cv2.imshow('MultiVision', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()