import os
import face_recognition
import pickle

def train_model(dataset_dir='face_detection/dataset', model_file='face_detection/face_training.pkl'):
    known_face_encodings = []
    known_face_names = []
    
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)
                    print(f"Encoded face for {person_name} from {filename}")
                else:
                    print(f"No face found in {filename} of {person_name}")
    
    with open(model_file, 'wb') as f:
        pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
    print(f"Model saved to {model_file}")
