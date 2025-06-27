from OCR_CCCD import extract_info_from_image
from Face_reg import load_encodings
import os


if __name__ == "__main__":
    while True:
        print("-"* 50)
        print("Thêm mới thông tin người mới: 1")
        print("Định danh khuôn mặt: 2")
        print("Thoát chương trình: q ")
        print("-"* 50)
        choice  = input("Nhập lựa chọn của bạn: ")
        if choice == '1':
            image_path = input("Nhập đường dẫn đến ảnh CCCD: ")
            if not image_path or not os.path.exists(image_path):
                print("Vui lòng nhập đường dẫn ảnh.")
                continue
            print("Đang trích xuất thông tin từ CCCD...")
            result = extract_info_from_image(image_path)
            print("Đã cập nhật người mới vào hệ thống.")
        elif choice == '2':
            result = load_encodings()
    
        elif choice.lower() == 'q':
            print("Thoát chương trình.")
            exit(0)
