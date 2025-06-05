import cv2
import numpy as np
from keras.models import load_model
import os
from datetime import datetime
import csv

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model("facialemotionmodel_finetuned_v2.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_model.yml")

label_map = {}
with open("labels.txt", "r") as f:
    for line in f:
        label_id, name = line.strip().split(",")
        label_map[int(label_id)] = name

log_file = "emotion_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Name", "Emotion", "Confidence"])

cap = cv2.VideoCapture(0)
print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(roi_gray, (48, 48)).reshape(1, 48, 48, 1).astype("float32") / 255.0
        emotion_idx = np.argmax(emotion_model.predict(face_resized, verbose=0))
        predicted_emotion = emotion_labels[emotion_idx]

        roi_resized_for_recog = cv2.resize(roi_gray, (200, 200))
        label_id, confidence = face_recognizer.predict(roi_resized_for_recog)
        name = label_map.get(label_id, "Unknown")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, name, predicted_emotion, round(confidence, 2)])

        overlay = f"{name} | {predicted_emotion} ({round(confidence, 1)})"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 180, 255), 2)
        cv2.rectangle(frame, (x, y-35), (x+w, y), (40, 180, 255), -1)
        cv2.putText(frame, overlay, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Face + Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
