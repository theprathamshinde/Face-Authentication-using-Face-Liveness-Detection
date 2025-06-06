import os
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import util
from FaceLiveness.test import test
import matplotlib.pyplot as plt
import numpy as np

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        # Instructions label with black font
        self.instructions_label = tk.Label(
            self.main_window, text="Instructions:\n1. The environment behind the face must not reflect light.\n2. The face must not be too far from the camera.",
            font=("Arial", 12), fg="black", justify="left"
        )
        self.instructions_label.place(x=750, y=50)

        self.login_button_main_window = util.get_button(self.main_window, 'Submit', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, 'Register New User', 'gray', self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

    def add_webcam(self, label):
        self.cap = None
        for index in range(5):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                self.cap = cap
                print(f"Camera found at index {index}")
                break
            cap.release()

        if not self.cap or not self.cap.isOpened():
            print("Error: Could not access the camera.")
            tk.Label(self.main_window, text="No camera detected!", fg="red", font=("Arial", 20)).place(x=100, y=200)
            return

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame from camera.")
            return

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):
        if not hasattr(self, "most_recent_capture_arr"):
            util.msg_box("Error", "No frame captured from webcam.")
            return

        # Run liveness test to detect spoofing
        liveness_score = test(
            image=self.most_recent_capture_arr,
            model_dir='FaceLiveness/resources/anti_spoof_models',
            device_id=0
        )

        if liveness_score == 1:
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)
            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Welcome back !', f'Welcome, {name}.')
                with open(self.log_path, 'a') as f:
                    f.write(f'{name},{datetime.datetime.now()},in\n')
                self.plot_face_comparison(name, liveness_score)
        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake!')


    def plot_face_comparison(self, name, liveness_score):
        # Load the registered face encoding from the database
        db_encoding_path = os.path.join(self.db_dir, f'{name}.pickle')
        if not os.path.exists(db_encoding_path):
            util.msg_box("Error", "Registered face not found in database.")
            return

        with open(db_encoding_path, 'rb') as f:
            db_encoding = pickle.load(f)

        # Get the encoding for the captured face from the webcam
        webcam_encoding = face_recognition.face_encodings(self.most_recent_capture_arr)[0]

        # Compare the two encodings (database vs. captured face)
        face_distance = face_recognition.face_distance([db_encoding], webcam_encoding)[0]
        similarity_score = max(0, 1 - face_distance)  # Convert distance to similarity (1 is perfect match)

        # Generate an arbitrary confidence score (for example, based on the similarity score)
        confidence_score = similarity_score * 0.8 + liveness_score * 0.2  # Arbitrary weighted score

        # Generate face detection confidence (this could be derived from how well the face is detected)
        face_locations = face_recognition.face_locations(self.most_recent_capture_arr)
        face_detection_confidence = len(face_locations) / 1  # A simple detection confidence (1 if face is found)

        # Prepare data for plotting (line graph)
        metrics = ['Face Recognition', 'Liveness', 'Confidence', 'Face Detection']
        scores = [similarity_score, liveness_score, confidence_score, face_detection_confidence]

        # Plot the line graph with annotations
        plt.figure(figsize=(8, 6))
        plt.plot(metrics, scores, marker='o', color='blue', label='Scores')

        # Annotating each point with the value in black
        for i, score in enumerate(scores):
            plt.text(metrics[i], score + 0.05, f'{score:.2f}', ha='center', color='black')  # Added some spacing

        plt.title(f"Face Recognition, Liveness, Confidence, and Detection - {name}", pad=30)  # Added padding for spacing after the title
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()

        # Popup message with the similarity score and liveness status
        util.msg_box('Face Recognition, Liveness, Confidence, and Detection Results',
                     f"Similarity Score: {similarity_score:.2f}\nLiveness Score: {'Real' if liveness_score == 1 else 'Fake'}\n"
                     f"Confidence Score: {confidence_score:.2f}\nFace Detection Confidence: {face_detection_confidence:.2f}")

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user
        )
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user
        )
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Please, \ninput username:'
        )
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]

        file = open(os.path.join(self.db_dir, f'{name}.pickle'), 'wb')
        pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User was registered successfully!')
        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()