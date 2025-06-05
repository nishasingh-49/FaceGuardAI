import cv2
import os
import numpy as np
data_dir = "training_data"
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
label_map = {}
current_label = 0

print("[INFO] Scanning training_data folder...")
for person_name in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_name)
    if not os.path.isdir(person_path):
        continue
    print(f"[INFO] Processing '{person_name}'...")
    label_map[current_label] = person_name
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Skipping unreadable image: {img_path}")
            continue
        img_resized = cv2.resize(img, (200, 200))
        faces.append(img_resized)
        labels.append(current_label)
    current_label += 1
print(f"[INFO] Training on {len(faces)} images...")
face_recognizer.train(faces, np.array(labels))
face_recognizer.save("face_model.yml")
print("[INFO] Training complete! Model saved to face_model.yml")
with open("labels.txt", "w") as f:
    for label_id, name in label_map.items():
        f.write(f"{label_id},{name}\n")
