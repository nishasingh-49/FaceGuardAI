import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import sqlite3
from datetime import datetime, date
import smtplib
from email.mime.text import MIMEText


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
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        face_label INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS alert_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        face_label INTEGER,
        alert_date TEXT
    )
""")
conn.commit()
conn.close()

def get_user_info(face_label):
    conn = sqlite3.connect("emotion_log.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, email FROM users WHERE face_label = ?", (face_label,))
    result = cursor.fetchone()
    conn.close()
    return result

def has_already_alerted_today(face_label):
    today = str(date.today())
    conn = sqlite3.connect("emotion_log.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alert_log WHERE face_label = ? AND alert_date = ?", (face_label, today))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def log_alert(face_label):
    today = str(date.today())
    conn = sqlite3.connect("emotion_log.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO alert_log (face_label, alert_date) VALUES (?, ?)", (face_label, today))
    conn.commit()
    conn.close()

def send_warning_email(to_email, person_name):
    subject = "\u26a0\ufe0f Helmet Violation Detected"
    body = f"""
    Hello {person_name},

    Our smart surveillance system detected that you were not wearing a helmet today.

    For your safety, please ensure you wear a helmet while riding.

    Stay safe,
    Team BubAI 😎🛡️
    """
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "your_email@gmail.com"
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login("your_email@gmail.com", "your_app_password")
            server.send_message(msg)
        print(f"Alert email sent to {person_name} at {to_email}")
    except Exception as e:
        print("Failed to send email:", e)

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

    cv2.putText(frame, dt_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
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
        cv2.putText(frame, label_text, (x_full + 5, y_full - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if not helmet_detected:
            warning_text = "Helmet Not detected, please wear it!"
            cv2.putText(frame, warning_text, (x_full + 5, y_full + h_full + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if name != "Unknown":
                try:
                    label_id = list(label_map.keys())[list(label_map.values()).index(name)]
                    if not has_already_alerted_today(label_id):
                        user_info = get_user_info(label_id)
                        if user_info:
                            send_warning_email(user_info[1], user_info[0])
                            log_alert(label_id)
                except Exception as e:
                    print("Error during alert:", e)

    cv2.imshow("Face + Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
