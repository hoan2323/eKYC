import tkinter as tk
from tkinter import Label, Frame
from PIL import Image, ImageTk
import cv2
import pickle
import numpy as np
import os
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import face_recognition

def Face_Recognition_and_Anti_Spoofing():
    # Load dữ liệu encoding
    def load_encodings():
        with open("face_encodings_1.pkl", "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]

    known_encodings, known_names = load_encodings()

    # Tải mô hình chống giả mạo
    model = tf.keras.models.load_model('best_model_retrained_1.h5')

    # Tải Haar Cascade để phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Không thể tải Haar Cascade!")

    # Hàm tiền xử lý ảnh khuôn mặt cho mô hình chống giả mạo
    def preprocess_face(face_img, img_height=224, img_width=224):
        face_img = cv2.resize(face_img, (img_height, img_width))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = img_to_array(face_img) / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    # Hàm dự đoán thật/giả
    def predict_face(face_img):
        processed_face = preprocess_face(face_img)
        prediction = model.predict(processed_face, verbose=0)[0][0]
        label = 'Fake' if prediction > 0.5 else 'Real'
        confidence = prediction if prediction > 0.5 else 1 - prediction
        return label, confidence

    # Hàm tìm và đọc file .csv từ thư mục có tên folder_name
    def read_info_csv(folder_name):
        for root, dirs, files in os.walk("."):
            if os.path.basename(root) == folder_name:
                for file in files:
                    if file.endswith(".csv"):
                        with open(os.path.join(root, file), encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            return next(reader, None)
        return None

    class FaceRecognitionApp:
        def __init__(self, window, window_title):
            self.window = window
            self.window.title(window_title)
            
            # Khung chính
            self.main_frame = Frame(window)
            self.main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Khung hiển thị video
            self.video_frame = Frame(self.main_frame)
            self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Label để hiển thị video
            self.video_label = Label(self.video_frame)
            self.video_label.pack()
            
            # Khung hiển thị thông tin
            self.info_frame = Frame(self.main_frame, width=300)
            self.info_frame.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Label để hiển thị thông tin
            self.info_label = Label(self.info_frame, text="Thông tin cá nhân:", font=("Arial", 14))
            self.info_label.pack(pady=10)
            
            # Ô hiển thị thông tin chi tiết
            self.info_text = tk.Text(self.info_frame, height=20, width=50)
            self.info_text.pack(pady=10)
            
            # Mở webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Không mở được webcam.")
                exit()
            
            # Bắt đầu cập nhật video
            self.update_video()
            
        def update_video(self):
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                # Biến cờ để theo dõi xem có khuôn mặt đúng hay không
                face_detected_correctly = False
                
                # Xóa ô thông tin trước khi xử lý khung hình mới
                self.info_text.delete(1.0, tk.END)
                
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    
                    try:
                        label, confidence = predict_face(face_img)
                        if label == 'Fake':
                            color = (0, 0, 255)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, 'The_face_was_spoof', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        else:
                            small_face = cv2.resize(face_img, (0, 0), fx=0.25, fy=0.25)
                            rgb_small_face = small_face[:, :, ::-1]
                            face_locations = [(0, small_face.shape[1], small_face.shape[0], 0)]
                            face_encodings = face_recognition.face_encodings(rgb_small_face, face_locations)
                            
                            if face_encodings:
                                face_encoding = face_encodings[0]
                                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                                name = "Unknown"
                                
                                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)
                                
                                if matches[best_match_index]:
                                    name = known_names[best_match_index]
                                    match_score = 1 - face_distances[best_match_index]
                                    
                                    if match_score > 0.6:
                                        face_detected_correctly = True
                                        info = read_info_csv(name)
                                        if info:
                                            self.info_text.insert(tk.END, f"Thông tin của {name}:\n")
                                            for k, v in info.items():
                                                self.info_text.insert(tk.END, f"{k}: {v}\n")
                                        else:
                                            self.info_text.insert(tk.END, f"Không tìm thấy thông tin cho {name}")
                                        
                                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    else:
                                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                        cv2.putText(frame, "No_infomation", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"Lỗi xử lý khuôn mặt: {e}")
                        continue
                
                # Chuyển frame sang định dạng ImageTk
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            
            # Lặp lại sau 10ms
            self.window.after(10, self.update_video)

    # Tạo cửa sổ và ứng dụng
    root = tk.Tk()
    app = FaceRecognitionApp(root, "Face Recognition and Anti-Spoofing")
    root.mainloop()