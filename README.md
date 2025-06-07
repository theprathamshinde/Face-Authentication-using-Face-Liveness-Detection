# 👁️‍🗨️ Face Authentication using Face Liveness Detection

This project implements a real-time **face authentication system** with **liveness detection** to prevent spoofing attacks using photos, videos, or masks. The system captures and processes webcam input to determine whether the face presented is live and genuine, thereby enhancing security in user authentication.

---

## 🧩 Problem Statement

Traditional face recognition systems are vulnerable to spoofing attacks using printed photos, replayed videos, or 3D masks. This project aims to build a lightweight, efficient, and accurate system that can **detect liveness** and distinguish between a live person and a spoof attempt using a **standard webcam**.

---

## 🎯 Objectives

1. ✅ **Develop a Real-Time Face Liveness Detection System**  
   Implement a model capable of distinguishing between live and fake faces in real time using webcam feed.

2. 🛡️ **Enhance Security by Preventing Spoofing Attempts**  
   Incorporate advanced image-based detection techniques to block photo, video, or mask-based attacks.

3. 🙋‍♂️ **Enable Seamless User Recognition and Registration**  
   Allow genuine users to register and authenticate smoothly without friction.

---

## 🧪 Methodology

- **Face Detection**: Use a pre-trained face detector (e.g., Haar Cascade or MTCNN) to localize the face in video frames.
- **Liveness Detection**: Use deep learning (CNN-based) classifier to detect liveness from facial features.
- **User Authentication**: Match live faces with registered identities using a face embedding model (e.g., FaceNet, Dlib).
- **Spoof Prevention**: Real-time rejection of spoofed inputs based on liveness confidence score.

---

## 🧰 Tech Stack

- **Language**: Python
- **Libraries**:
  - OpenCV (Webcam input, frame processing)
  - TensorFlow / PyTorch (Liveness model)
  - Dlib / FaceNet (Face embedding & recognition)
  - NumPy, scikit-learn (Data processing)
- **Hardware**: Standard webcam (no special sensors required)

---

## 📁 Project Structure



Project Demo Video : https://drive.google.com/drive/folders/1u_8F_7JbJqvZLAW_N-W1pBTaNy2qmQKx?usp=drive_link

Architectural Design
![image](https://github.com/user-attachments/assets/6681bae2-70be-4a42-bded-52396373f57d)

![image](https://github.com/user-attachments/assets/50b4451f-c7e1-4829-a998-5d9b4d407ff5)

