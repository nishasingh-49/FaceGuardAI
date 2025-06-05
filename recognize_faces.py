import cv2

# Load model and labels
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_model.yml")

label_map = {}
with open("labels.txt", "r") as f:
    for line in f:
        label_id, name = line.strip().split(",")
        label_map[int(label_id)] = name

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label_id, confidence = face_recognizer.predict(roi_gray)
        name = label_map.get(label_id, "Unknown")
        text = f"{name} ({round(confidence, 2)})"

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("OpenCV Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
