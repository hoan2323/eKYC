🔐 eKYC System 
![Screenshot 2025-07-05 151855](https://github.com/user-attachments/assets/3eddc94c-f27a-4df7-aa72-e6256a4544ad)
![Thiết kế chưa có tên (1)](https://github.com/user-attachments/assets/c4ca7696-861d-43f1-ad0b-984728339790)
![Thiết kế chưa có tên](https://github.com/user-attachments/assets/283755e5-8c9b-4254-9156-dc592e3df3ae)


🧠 1. train_CNNmodel/ – Training the Face Anti-Spoofing Model

Data Collection and Preprocessing:

- Balance the dataset between real and spoofed samples.

- Crop face regions from images to focus on relevant areas and remove background noise, enhancing model performance.

Model Training:

- Train a lightweight yet efficient MobileNetV2 model for real-time face spoof detection.

- Save the model in .h5 formats for easy deployment.

🧾 2. eKYC_main/ – Core eKYC System
This module handles the full pipeline of identity verification:

📄 CCCD OCR

- Uses 5CD-AI/Vintern-1B-v3_5, a Visual Language Model (VLM), to:

- Extract personal information from Vietnamese ID cards (CCCD).

🧍‍♂️ Face Recognition
- Compares the face from the live camera with the face on the ID card.

- Ensures the person presenting the ID matches the photo on it.

🛡️ Face Anti-Spoofing
- Integrates the anti-spoofing model trained in train_CNNmodel/:

- Detects attempts to fool the system using printed photos or video replays.

- Improves the overall security and trustworthiness of the eKYC process.
