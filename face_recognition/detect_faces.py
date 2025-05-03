import face_recognition
import pickle
import cv2

def load_model(model_file='face_recognition/face_training.pkl'):
    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        return data['encodings'], data['names']
    except FileNotFoundError:
        print(f"Warning: Model file '{model_file}' not found.")
        return [], []
    except Exception as e:
        print(f"Error loading model: {e}")
        return [], []

def recognize_faces_from_webcam(model_file='face_training.pkl'):
    known_encodings, known_names = load_model(model_file)
    
    video_capture = cv2.VideoCapture(0)          

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit.")
    while True:

        ret, frame = video_capture.read()
        
        if not ret or frame is None:
            print("Warning: Failed to capture frame from webcam. Retrying...")
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if known_encodings:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                if True in matches:
                    name = known_names[matches.index(True)]
                else:
                    name = "Unknown"
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Webcam Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

recognize_faces_from_webcam()