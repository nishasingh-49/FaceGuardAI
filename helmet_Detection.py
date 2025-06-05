import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import sqlite3
from datetime import datetime
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_model.yml")
emotion_model = load_model("facialemotionmodel_finetuned_v2.keras")
label_map = {}
with open("labels.txt", "r") as f:
    for line in f:
        label_id, name = line.strip().split(",")
        label_map[int(label_id)] = name
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
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
def get_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning! Have a great day!"
    elif 12 <= hour < 17:
        return "Good Afternoon! Keep going!"
    elif 17 <= hour < 21:
        return "Good Evening! Hope you had a good day!"
    else:
        return "Good Night! Take care!"
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
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    greeting = get_greeting()
    (dt_w, dt_h), _ = cv2.getTextSize(dt_string, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (5, 10), (5 + dt_w + 10, 10 + dt_h + 10), (0, 0, 0), -1)
    cv2.putText(frame, dt_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    (gr_w, gr_h), _ = cv2.getTextSize(greeting, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (5, 40), (5 + gr_w + 10, 40 + gr_h + 10), (0, 0, 0), -1)
    cv2.putText(frame, greeting, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for i, (x, y, w, h) in enumerate(faces):
        x_full, y_full, w_full, h_full = x * 2, y * 2, w * 2, h * 2
        roi_gray = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        helmet_roi = gray[max(y - int(h / 1.5), 0):y, x:x + w]
        avg_intensity = np.mean(helmet_roi) if helmet_roi.size > 0 else 255
        helmet_detected = avg_intensity < 80
        cache_key = f"face_{i}"
        if frame_count % 5 == 0 or cache_key not in last_results:
            try:
                label_id, confidence = face_recognizer.predict(roi_gray)
                name = label_map.get(label_id, "Unknown")
            except:
                name = "Unknown"
                confidence = 0.0
            try:
                emotion_roi = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
                emotion_roi = emotion_roi.astype("float32") / 255.0
                emotion_roi = img_to_array(emotion_roi)
                emotion_roi = np.expand_dims(emotion_roi, axis=0)
                preds = emotion_model.predict(emotion_roi, verbose=0)[0]
                emotion = emotion_labels[np.argmax(preds)]
            except:
                emotion = "Unknown"
            last_results[cache_key] = (name, emotion)
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
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
                print("Failed to log to SQLite:", e)
        else:
            name, emotion = last_results[cache_key]
        label_text = f"{name} | {emotion}"
        cv2.rectangle(frame, (x_full, y_full), (x_full + w_full, y_full + h_full), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x_full, y_full - th - 10), (x_full + tw + 10, y_full), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (x_full + 5, y_full - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if not helmet_detected:
            warning_text = "Helmet Not detected,please wear it!"
            (wtw, wth), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x_full, y_full + h_full + 10), (x_full + wtw + 10, y_full + h_full + wth + 20), (0, 0, 0), -1)
            cv2.putText(frame, warning_text, (x_full + 5, y_full + h_full + wth + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Face + Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
