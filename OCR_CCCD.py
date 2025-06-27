import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
# Ensure protobuf is downgraded before any imports that use it
# %pip install protobuf==3.20.3
import os
import warnings
import json
import cv2
import time
from unidecode import unidecode
import csv
import face_recognition
import pickle

# Tắt warning của TensorFlow (cudart missing, v.v.)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Tắt tất cả cảnh báo Python (FutureWarning, UserWarning, ...)
warnings.filterwarnings("ignore")

# Tắt log của transformers
from transformers.utils import logging
logging.set_verbosity_error()

# Tắt cảnh báo của flash attention nếu không cài đặt
import warnings
warnings.filterwarnings("ignore", message="FlashAttention2 is not installed.")



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

model = AutoModel.from_pretrained(
    "5CD-AI/Vintern-1B-v3_5",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_flash_attn=False,
).eval().cpu()

tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v3_5", trust_remote_code=True, use_fast=False)


def extract_info_from_image(image_path):
    pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).cpu()
    question = '''<image>\nTrích xuất thông tin chính trong ảnh theo Họ và tên,'Số thẻ, Ngày sinh, Giới tính, Quốc tịch, Quê quán, Nơi thường trú, Có giá trị đến và trả về định dạng json như sau:
json
{
  "Họ và tên": "...",
  "Số thẻ": "",
  "Ngày sinh": "...",
  "Giới tính": "...",
  "Quốc tịch": "...",
  "Quê quán": "....",
  "Nơi thường trú": "...",
  "Có giá trị đến": "..."
}
 '''
    response, history = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config=dict(max_new_tokens=1024, do_sample=False, num_beams=3, repetition_penalty=2.5),
        history=None,
        return_history=True
    )
    response_cleaned = response.strip().strip('`').strip()
    if response_cleaned.startswith("json"):
        response_cleaned = response_cleaned[4:].strip()
    
    info = json.loads(response_cleaned)
    print("-"*50)
    print("Thông tin CCCD của bạn\n", info)
    print("-"*50)
    print("Kiểm tra thông tin CCCD, nếu đúng thì nhập 'y' để tiếp tục, nếu không thì nhập 'n':")
    task_ocr = input("Nhập lựa chọn của bạn (y/n): ").strip().lower()
    if task_ocr == 'y':
        person_name_raw = info["Họ và tên"]
        person_name = unidecode(person_name_raw.strip().replace(" ", "_"))
        folder_name = f"{info['Số thẻ']}_{person_name}"
        full_path = os.path.join(folder_name) 

        # ==== Tạo folder chính nếu chưa có ====
        os.makedirs(full_path, exist_ok=True)

        # ==== Tạo folder con trong full_path với tên giống folder chính ====
        subfolder_path = os.path.join(full_path, folder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # ==== Ghi thông tin vào CSV trong folder con ====
        info_file = os.path.join(subfolder_path, f"info_{info['Số thẻ']}_{person_name_raw}.csv")
        with open(info_file, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Họ và tên", "Số thẻ", "Ngày sinh", "Giới tính", "Quốc tịch", "Quê quán","Nơi thường trú", "Có giá trị đến"])
            writer.writerow([
                info['Họ và tên'],
                info['Số thẻ'],
                info['Ngày sinh'],
                info['Giới tính'],
                info['Quốc tịch'],
                info['Quê quán'],  
                info['Nơi thường trú'],  
                info['Có giá trị đến']
            ])
        print("Thực hiện ghi trích xuất khuân mặt")
        time.sleep(3)
        # ==== Mở webcam để chụp ảnh và lưu trong folder con ====
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không mở được webcam.")
            exit()

        count = 0
        max_images = 200

        while count < max_images:
            ret, frame = cap.read()
            if not ret:
                print("Không thể lấy ảnh từ webcam.")
                break

            frame = cv2.flip(frame, 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            img_name = f"{count}.jpg"
            save_path = os.path.join(subfolder_path, img_name)
            if cv2.imwrite(save_path, gray_frame):
                print(f"Đã lưu khuôn mặt số {img_name}")
            else:
                print(f"Lỗi lưu {img_name}")

            cv2.imshow("Face dataset", frame)
            count += 1
            

        cap.release()
        cv2.destroyAllWindows()
        # ==== Lưu các encoding khuôn mặt vào file .pkl ====
        encodings = []
        names = []
        pkl_file = "face_encodings.pkl"

        # ==== Load dữ liệu đã có nếu file .pkl tồn tại ====
        if os.path.exists(pkl_file):
            with open(pkl_file, "rb") as f:
                existing_data = pickle.load(f)
                encodings = existing_data.get("encodings", [])
                names = existing_data.get("names", [])
                print(f"Đã nạp {len(encodings)} khuôn mặt đã lưu.")

        # ==== Lặp qua các thư mục trong dataset mới ====
        for folder_name in os.listdir(full_path):
            person_dir = os.path.join(full_path, folder_name)
            if not os.path.isdir(person_dir):
                continue

            print(f"Đang xử lý: {folder_name}")

            for file_name in os.listdir(person_dir):
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(person_dir, file_name)

                # Đọc ảnh và nhận diện khuôn mặt
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image)

                if len(face_locations) == 0:
                    print(f"Không tìm thấy khuôn mặt trong {file_name}")
                    continue

                # Nếu đã có encoding của người này và ảnh này thì bỏ qua (tuỳ chọn)
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                encodings.append(face_encoding)
                names.append(folder_name)
                print(f"Đã lưu encoding cho {file_name} trong {folder_name}")

        # ==== Ghi lại toàn bộ encodings + names vào file .pkl ====
        data = {"encodings": encodings, "names": names}
        with open(pkl_file, "wb") as f:
            pickle.dump(data, f)

        print(f"Đã cập nhật {len(encodings)} khuôn mặt vào: {pkl_file}")
        with open("face_encodings.pkl", "rb") as f:
            data = pickle.load(f)
            print("Số khuôn mặt đã lưu:", len(data["encodings"]))

