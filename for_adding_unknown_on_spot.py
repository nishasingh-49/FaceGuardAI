import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import sqlite3
from datetime import datetime
import os
import tkinter as tk
from tkinter import simpledialog

# ====== Load Models ======
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_model.yml")
emotion_model = load_model("facialemotionmodel_finetuned_v2.keras")

# ====== Label Map ======
label_map = {}
if os.path.exists("labels.txt"):
    with open("labels.txt", "r") as f:
        for line in f:
            label_id, name = line.strip().split(",")
            label_map[int(label_id)] = name

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ====== SQLite Setup ======
conn = sqlite3.connect("emotion_log.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS emotion_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        name TEXT,   
        emotion TEXT,
        confidence REAL
    )
""")
conn.commit()
conn.close()

# ====== Tkinter Setup ======
root = tk.Tk()
root.withdraw()  # Hide root window

# ====== Webcam Setup ======
cap = cv2.VideoCapture(0)
frame_count = 0
last_results = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    frame_count += 1

    for i, (x, y, w, h) in enumerate(faces):
        x_full, y_full, w_full, h_full = x*2, y*2, w*2, h*2
        roi_gray = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

        cache_key = f"face_{i}"
        if frame_count % 5 == 0 or cache_key not in last_results:
            try:
                label_id, confidence = face_recognizer.predict(roi_gray)
                name = label_map.get(label_id, "Unknown")
                if confidence > 60:
                    name = "Unknown"
            except:
                name = "Unknown"
                confidence = 0.0

            # 🧠 Emotion Detection
            try:
                emotion_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                emotion_roi = emotion_roi.astype("float32") / 255.0
                emotion_roi = img_to_array(emotion_roi)
                emotion_roi = np.expand_dims(emotion_roi, axis=0)
                preds = emotion_model.predict(emotion_roi, verbose=0)[0]
                emotion = emotion_labels[np.argmax(preds)]
            except:
                emotion = "Unknown"

            # 👤 Unknown Face Handling
            if name == "Unknown":
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                face_crop = frame[y_full:y_full+h_full, x_full:x_full+w_full]
                cv2.imshow("Unknown Face", face_crop)
                cv2.waitKey(1)

                # 🔤 Ask for label using popup
                user_input = simpledialog.askstring("Unknown Face Detected", "Enter name to label this face:")
                if user_input:
                    label_dir = os.path.join("dataset", user_input)
                    os.makedirs(label_dir, exist_ok=True)
                    filename = os.path.join(label_dir, f"{timestamp_str}.jpg")
                    cv2.imwrite(filename, face_crop)

                    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    gray_face_resized = cv2.resize(gray_face, (200, 200))

                    # 🔢 Assign label ID
                    if user_input in label_map.values():
                        label_id = [k for k, v in label_map.items() if v == user_input][0]
                    else:
                        label_id = max(label_map.keys(), default=0) + 1
                        label_map[label_id] = user_input
                        with open("labels.txt", "a") as f:
                            f.write(f"{label_id},{user_input}\n")

                    # 🔁 Update recognizer
                    face_recognizer.update([gray_face_resized], np.array([label_id]))
                    name = user_input
                    confidence = 0.0
                else:
                    print("⏭️ No label entered.")
                cv2.destroyWindow("Unknown Face")

            last_results[cache_key] = (name, emotion)

            # 🗂️ Log to SQLite
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                conn = sqlite3.connect("emotion_log.db")
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO emotion_logs (timestamp, name, emotion, confidence)
                    VALUES (?, ?, ?, ?)
                """, (timestamp, name, emotion, float(confidence)))
                conn.commit()
                conn.close()
            except Exception as e:
                print("❌ Failed to log to SQLite:", e)

        else:
            name, emotion = last_results[cache_key]

        # 📌 Draw on frame
        label_text = f"{name} | {emotion}"
        cv2.rectangle(frame, (x_full, y_full), (x_full + w_full, y_full + h_full), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x_full, y_full - th - 10), (x_full + tw + 10, y_full), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (x_full + 5, y_full - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face + Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
