import os
import pickle
import cv2
import face_recognition
from pathlib import Path

TRAINING_DIR = "dataset"
DATA_DIR = "data"
BATCH_SIZE = 50

known_encodings = []
known_names = []

for person_id in os.listdir(TRAINING_DIR):
    person_dir = os.path.join(TRAINING_DIR, person_id)
    if not os.path.isdir(person_dir):
        continue
    image_paths = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        for image_path in batch_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(person_id)

        if len(known_encodings) >= BATCH_SIZE:
            with open(os.path.join(DATA_DIR, "encodings.pkl"), "wb") as f:
                pickle.dump({"encodings": known_encodings, "names": known_names}, f)
            print(f"Saved checkpoint with {len(known_encodings)} encodings.")

with open(os.path.join(DATA_DIR, "encodings.pkl"), "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print("Training complete. Encodings saved to data/encodings.pkl.")
