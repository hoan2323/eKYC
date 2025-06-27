import face_recognition
import cv2
import pickle
import numpy as np
import os
import csv

# Load dữ liệu encoding
def load_encodings():
    with open("face_encodings_1.pkl", "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam.")
        exit()

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

    recognized_names = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ webcam.")
            break

        frame = cv2.flip(frame, 1)

        # Thu nhỏ ảnh để xử lý nhanh
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Chuyển BGR -> RGB

        # Tìm khuôn mặt
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if matches[best_match_index]:
                name = known_names[best_match_index]
                match_score = 1 - face_distances[best_match_index]

                if match_score > 0.6:
                    if name not in recognized_names:
                        recognized_names.add(name)
                        info = read_info_csv(name)
                        if info:
                            print(f"== Thông tin của {name} ==")
                            for k, v in info.items():
                                print(f"{k}: {v}")
                        else:
                            print(f"Không tìm thấy file info cho {name}")

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    info_to_scereen  =  f"{name} ({match_score:.2f})" 
                    cv2.putText(frame, info_to_scereen, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Không có thông tin", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Demo ekyc", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
            break

    cap.release()
    cv2.destroyAllWindows()




    
