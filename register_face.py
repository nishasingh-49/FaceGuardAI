import cv2
import os

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ask for user name
name = input("Enter your name for registration: ").strip().lower()
save_dir = os.path.join("training_data", name)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press 'q' to quit early.")

count = 0
max_images = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (200, 200))
        count += 1
        cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), face_resized)

        # Draw rectangle and show count
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Image {count}/{max_images}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] {count} images saved to {save_dir}.")
print("[INFO] Now run `train_model.py` to update the face recognizer.")
