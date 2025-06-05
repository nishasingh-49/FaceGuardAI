# FaceGuardAI
FaceGuardAI is a real-time facial recognition and emotion-aware helmet detection system. Combines advanced AI vision with public safety intelligence—logging emotional states and helmet compliance with live alerts and SQLite logging. Future-ready with email alert integration.<br>
> Real-time Face Recognition · Emotion Detection · Helmet Violation Alerts  
> Because safety starts with seeing more than just a face.

---
**FaceGuardAI** is an intelligent surveillance system that:
- Identifies individuals via **face recognition**
- Detects **emotional states** using a trained deep learning model
- Checks for **helmet compliance** (“yo where your helmet at?”)
- Issues **on-screen alerts**
- Logs all detections (name, emotion, confidence, timestamp) in **SQLite**
- Ready for future **email alert integration** for violations

This project is perfect for workplace safety, public road monitoring, or any AI-enhanced awareness tool. 

---

# Features

-Real-time **face recognition** (OpenCV + LBPH)  
-Accurate **emotion detection** (TensorFlow/Keras CNN model)  
-Helmet detection based on ROI brightness  
-Visual alerts for helmet violations  
-Emotion + identity **logging into SQLite**  
-Timestamp and dynamic **greeting UI**  
-Email alert logic (coming soon!)  

---

# Tech Stack

- Python 
- OpenCV 
- TensorFlow / Keras 
- NumPy
- SQLite
- [Twilio](https://www.twilio.com/) *( for future SMS/Email alerts)*

# Future Enhancements
-Email alerts when helmet is not detected<br>
-Dashboard for visualizing logs (Chart.js + Flask)<br>
-Admin panel to manage users, track violators<br>
-Cloud deployment with video stream input<br>

# get the dataset from kaggle or any other online platform!!
