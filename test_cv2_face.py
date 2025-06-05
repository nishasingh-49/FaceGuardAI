import cv2

print("OpenCV version:", cv2.__version__)
recognizer = cv2.face.LBPHFaceRecognizer_create()
print("LBPH Recognizer loaded successfully!")
