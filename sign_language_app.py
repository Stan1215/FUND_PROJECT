import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import pickle
import csv
import os
import pyttsx3
import urllib.request
import matplotlib.pyplot as plt
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_FILE = 'hand_landmarks_data.csv'
MODEL_FILE = 'model.p'
TASK_FILE = 'hand_landmarker.task'
IMAGE_FOLDER = 'dataset_images'

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

PINK_COLOR = (180, 105, 255)
LIGHT_BLUE = (250, 206, 135)


def speak_text(text_to_speak):
    def speech_worker(text):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")

    threading.Thread(target=speech_worker, args=(text_to_speak,), daemon=True).start()


def download_task_file_if_missing():
    if not os.path.exists(TASK_FILE):
        print("[INFO] Downloading 'hand_landmarker.task' model file from MediaPipe...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, TASK_FILE)
        print("[SUCCESS] Model file downloaded successfully!\n")


def initialize_landmarker():
    download_task_file_if_missing()
    base_options = python.BaseOptions(model_asset_path=TASK_FILE)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, hand_landmarks):
    h, w, _ = rgb_image.shape
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(rgb_image, coords[start_idx], coords[end_idx], PINK_COLOR, 2)

    for x, y in coords:
        cv2.circle(rgb_image, (x, y), 5, LIGHT_BLUE, -1)


def collect_data():
    print("\n--- DATA COLLECTION ---")
    
    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = []
            for i in range(21):
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            header.append('label')
            writer.writerow(header)

    label = input("Enter the Label/Letter to collect (e.g., A, B, C): ").strip().upper()
    try:
        samples_target = int(input("Enter number of samples to collect (default: 200): "))
    except ValueError:
        print("Invalid input! Using default target of 200 samples.")
        samples_target = 200

    label_img_dir = os.path.join(IMAGE_FOLDER, label)
    os.makedirs(label_img_dir, exist_ok=True)

    detector = initialize_landmarker()
    cap = cv2.VideoCapture(0)

    collected_count = 0
    collecting = False

    print("\n[INFO] Press 'S' in the camera window to START collecting.")
    print("[INFO] Press 'Q' to exit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam feed.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                draw_landmarks_on_image(frame, hand_landmarks)

                if collecting and collected_count < samples_target:
                    base_x = hand_landmarks[0].x
                    base_y = hand_landmarks[0].y
                    base_z = hand_landmarks[0].z

                    row = []
                    for lm in hand_landmarks:
                        row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                    
                    row.append(label)

                    with open(DATASET_FILE, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)

                    img_path = os.path.join(label_img_dir, f"sample_{collected_count + 1}.jpg")
                    cv2.imwrite(img_path, frame)

                    collected_count += 1

        cv2.putText(frame, f"Label: {label} | Samples: {collected_count}/{samples_target}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not collecting:
            cv2.putText(frame, "Press 'S' to START", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "SAVING DATA & PHOTOS...", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('s'), ord('S')):
            collecting = True
        elif key in (ord('q'), ord('Q')) or collected_count >= samples_target:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"[SUCCESS] Collected {collected_count} CSV rows and saved sample photos in '{label_img_dir}'.\n")


def display_sample_photos(df):
    unique_labels = df['label'].unique()
    num_labels = len(unique_labels)

    if num_labels == 0:
        return

    print("[INFO] Displaying sample image previews for each trained sign...")
    fig, axes = plt.subplots(1, num_labels, figsize=(4 * num_labels, 4))
    if num_labels == 1:
        axes = [axes]

    for ax, lbl in zip(axes, unique_labels):
        label_dir = os.path.join(IMAGE_FOLDER, lbl)
        img_path = os.path.join(label_dir, "sample_10.jpg")
        
        if not os.path.exists(img_path):
            img_path = os.path.join(label_dir, "sample_1.jpg")

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"Label: '{lbl}'", fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f"No image found\nfor '{lbl}'", ha='center', va='center')
            ax.set_title(f"Label: '{lbl}'", fontsize=12)
        
        ax.axis('off')

    plt.suptitle("Trained Dataset Sample Photos", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def train_model():
    print("\n--- MODEL TRAINING ---")
    
    if not os.path.exists(DATASET_FILE):
        print("[ERROR] No dataset found! Collect data first using Option 1.")
        return

    df = pd.read_csv(DATASET_FILE)
    if df.empty or len(df) < 2:
        print("[ERROR] Not enough data in CSV file. Collect more samples.")
        return

    X = df.drop(columns=['label'])
    y = df['label']

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    print(f"[SUCCESS] Trained model saved to '{MODEL_FILE}'!\n")

    display_sample_photos(df)


def realtime_predict():
    print("\n--- REAL-TIME PREDICTION ---")
    
    if not os.path.exists(MODEL_FILE):
        print("[ERROR] No trained model ('model.p') found! Train a model using Option 2 first.")
        return

    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    detector = initialize_landmarker()
    cap = cv2.VideoCapture(0)

    current_text = ""
    last_predicted = ""
    confirm_frames = 0
    CONFIRM_THRESHOLD = 15

    print("[INFO] Press 'S' to SPEAK the full constructed text.")
    print("[INFO] Press 'C' to clear the current text.")
    print("[INFO] Press 'Q' to exit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam feed.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        predicted_characters = []

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                draw_landmarks_on_image(frame, hand_landmarks)

                base_x = hand_landmarks[0].x
                base_y = hand_landmarks[0].y
                base_z = hand_landmarks[0].z

                landmarks_input = []
                for lm in hand_landmarks:
                    landmarks_input.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

                if len(landmarks_input) == 63:
                    prediction = model.predict([landmarks_input])
                    predicted_characters.append(str(prediction[0]))

            if predicted_characters:
                current_pred = predicted_characters[0]
                if current_pred == last_predicted:
                    confirm_frames += 1
                else:
                    last_predicted = current_pred
                    confirm_frames = 0

                if confirm_frames == CONFIRM_THRESHOLD:
                    current_text += current_pred
                    speak_text(current_pred)

        predicted_text = ", ".join(predicted_characters) if predicted_characters else "No hands detected"

        cv2.rectangle(frame, (0, 0), (640, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Prediction: {predicted_text}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if predicted_characters else (0, 0, 255), 2)
        cv2.putText(frame, f"Text: {current_text}", (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "[S] Speak All | [C] Clear | [Q] Quit", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Sign Language Translator", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('s'), ord('S')):
            if current_text:
                speak_text(current_text)
        elif key in (ord('c'), ord('C')):
            current_text = ""
        elif key in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


def delete_data_menu():
    print("\n--- DELETE / RESET DATASET ---")
    if not os.path.exists(DATASET_FILE):
        print("[INFO] No dataset file found.")
        return

    print("[1] Delete a SPECIFIC sign/letter (e.g., delete only 'A')")
    print("[2] Delete EVERYTHING (Full Reset)")
    print("[3] Cancel")

    choice = input("\nSelect an option (1-3): ").strip()

    if choice == '1':
        label_to_delete = input("Enter the letter to delete (e.g., A): ").strip().upper()
        df = pd.read_csv(DATASET_FILE)

        if label_to_delete in df['label'].values:
            df = df[df['label'] != label_to_delete]
            df.to_csv(DATASET_FILE, index=False)
            print(f"[SUCCESS] Deleted all CSV entries for letter '{label_to_delete}'.")
            print("[NOTE] Remember to re-train the model in Option 2 to update the system!")
        else:
            print(f"[INFO] No data found for letter '{label_to_delete}'.")

    elif choice == '2':
        confirm = input("Are you sure you want to DELETE ALL saved data? (Y/N): ").strip().upper()
        if confirm == 'Y':
            if os.path.exists(DATASET_FILE):
                os.remove(DATASET_FILE)
            if os.path.exists(MODEL_FILE):
                os.remove(MODEL_FILE)
            print("[SUCCESS] All dataset files and trained models have been deleted.")


def main():
    while True:
        print("========================================")
        print(" SIGN LANGUAGE TRANSLATOR (Random Forest)")
        print("========================================")
        print("[1] Collect Dataset (Save CSV + Image Snapshots)")
        print("[2] Train Model (Train Classifier & Show Photo Previews)")
        print("[3] Run Real-time Prediction (Webcam feed)")
        print("[4] Delete / Reset Dataset")
        print("[5] Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()

        if choice == '1':
            collect_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            realtime_predict()
        elif choice == '4':
            delete_data_menu()
        elif choice == '5':
            print("Exiting application. Goodbye!")
            break
        else:
            print("Invalid selection! Please enter a number between 1 and 5.\n")


if __name__ == "__main__":
    main()