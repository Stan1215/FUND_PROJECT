import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import pickle
import os
import pyttsx3
import urllib.request
import threading
from collections import Counter, deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

DATASET_FILE = "hand_landmarks_data.csv"
TEST_DATASET_FILE = "test_session_data.csv"
MODEL_FILE = "model.p"
TASK_FILE = "hand_landmarker.task"
MODEL_FEATURE_VERSION = 2
MIN_PREDICTION_CONFIDENCE = 0.7
MIN_PREDICTION_MARGIN = 0.1
PREDICTION_HISTORY_LENGTH = 8
MIN_STABLE_PREDICTIONS = 6
MIN_SAMPLES_PER_LABEL = 5  # absolute minimum to train; 40+ is recommended

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

PINK_COLOR = (180, 105, 255)
LIGHT_BLUE = (250, 206, 135)


# ---------------------------------------------------------------------------
# Speech and MediaPipe helpers
# ---------------------------------------------------------------------------
def speak_text(text_to_speak):
    def speech_worker(text):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
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
        base_options=base_options, num_hands=1, min_hand_detection_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, hand_landmarks):
    h, w, _ = rgb_image.shape
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(rgb_image, coords[start_idx], coords[end_idx], PINK_COLOR, 2)

    for x, y in coords:
        cv2.circle(rgb_image, (x, y), 5, LIGHT_BLUE, -1)


def select_primary_hand(hands):
    if not hands:
        return None

    def position_score(hand):
        xs = [landmark.x for landmark in hand]
        ys = [landmark.y for landmark in hand]
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        distance_from_center = (center_x - 0.5) * 2 + (center_y - 0.5) * 2
        return distance_from_center, -area

    return min(hands, key=position_score)


def normalize_landmarks(landmarks):
    values = np.array(landmarks, dtype=float, copy=True)
    if values.size != 63:
        raise ValueError("Expected 63 values for 21 hand landmarks.")

    values = values.reshape(21, 3)
    values -= values[0]
    palm_scale = float(np.linalg.norm(values[9]))
    if palm_scale < 1e-6:
        return None

    return (values / palm_scale).flatten().tolist()


# ---------------------------------------------------------------------------
# Dataset (CSV) helpers
# ---------------------------------------------------------------------------
def csv_header():
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")
    return header


def read_dataset(dataset_file):
    """Read a dataset CSV. Returns an empty DataFrame if missing or empty."""
    if not os.path.exists(dataset_file):
        return pd.DataFrame(columns=csv_header())
    try:
        return pd.read_csv(dataset_file)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=csv_header())


def count_label_rows(dataset_file, label):
    df = read_dataset(dataset_file)
    if df.empty:
        return 0
    return int((df["label"].astype(str) == label).sum())


def save_samples(dataset_file, label, new_rows, mode):
    """Write newly collected rows. mode is 'append' or 'replace'.

    'replace' removes every existing row of this label first, then adds the new
    rows. All other letters are never touched.
    """
    new_df = pd.DataFrame(new_rows, columns=csv_header())
    old_df = read_dataset(dataset_file)

    if mode == "replace" and not old_df.empty:
        old_df = old_df[old_df["label"].astype(str) != label]

    combined = (
        new_df if old_df.empty else pd.concat([old_df, new_df], ignore_index=True)
    )
    combined.to_csv(dataset_file, index=False)


def ask_existing_data_mode(label, existing_count):
    """Ask what to do when the letter already has saved samples."""
    print(f"\n[INFO] Letter '{label}' already has {existing_count} saved samples.")
    print("[A] Append  - keep the old samples and ADD the new ones")
    print(
        "[R] Replace - delete the old samples of this letter and use only the new ones"
    )
    print("[C] Cancel")
    while True:
        choice = input("Choose (A/R/C): ").strip().upper()
        if choice == "A":
            return "append"
        if choice == "R":
            return "replace"
        if choice == "C":
            return None
        print("Please type A, R, or C.")


def print_label_summary(y, title="Samples per letter"):
    counts = y.value_counts().sort_index()
    print(f"{title}: " + ", ".join(f"{lbl}={n}" for lbl, n in counts.items()))


# ---------------------------------------------------------------------------
# Data collection (no photos are saved - the model only needs landmark numbers)
# ---------------------------------------------------------------------------
def collect_data(dataset_file=DATASET_FILE, session_name="TRAINING DATA"):
    print(f"\n--- DATA COLLECTION: {session_name} ---")

    label = input("Enter the Label/Letter to collect (e.g., A, B, C): ").strip().upper()
    if not label:
        print("[ERROR] Label cannot be empty.\n")
        return

    mode = "append"
    existing_count = count_label_rows(dataset_file, label)
    if existing_count > 0:
        mode = ask_existing_data_mode(label, existing_count)
        if mode is None:
            print("[INFO] Cancelled. Nothing was changed.\n")
            return
        if mode == "replace":
            print(
                f"[INFO] Old '{label}' samples will only be replaced once new samples are saved."
            )

    try:
        samples_target = int(
            input("Enter number of samples to collect (default: 200): ")
        )
        if samples_target <= 0:
            raise ValueError
    except ValueError:
        print("Invalid input! Using default target of 200 samples.")
        samples_target = 200

    detector = initialize_landmarker()
    cap = cv2.VideoCapture(0)

    new_rows = []  # kept in memory, written to the CSV when collection ends
    collecting = False

    print("\n[INFO] Press 'S' in the camera window to START collecting.")
    print(
        "[INFO] Keep the signing hand near the center; only the closest hand is collected."
    )
    print("[INFO] Press 'Q' to stop and save what has been collected.\n")

    cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Data Collection", 1280, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam feed.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        hand_landmarks = select_primary_hand(detection_result.hand_landmarks)
        if hand_landmarks:
            draw_landmarks_on_image(frame, hand_landmarks)

            if collecting and len(new_rows) < samples_target:
                base_x = hand_landmarks[0].x
                base_y = hand_landmarks[0].y
                base_z = hand_landmarks[0].z

                row = []
                for lm in hand_landmarks:
                    row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                row.append(label)
                new_rows.append(row)

        cv2.putText(
            frame,
            f"{session_name} | Label: {label} | Samples: {len(new_rows)}/{samples_target}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        if not collecting:
            cv2.putText(
                frame,
                "Press 'S' to START",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "RECORDING LANDMARKS...",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("s"), ord("S")):
            collecting = True
        elif key in (ord("q"), ord("Q")) or len(new_rows) >= samples_target:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if not new_rows:
        print("[INFO] No samples were collected. Nothing was changed.\n")
        return

    save_samples(dataset_file, label, new_rows, mode)
    total_for_label = count_label_rows(dataset_file, label)
    action = "Replaced old data with" if mode == "replace" else "Appended"
    print(
        f"[SUCCESS] {action} {len(new_rows)} samples for '{label}' in '{dataset_file}'. "
        f"Letter '{label}' now has {total_for_label} samples.\n"
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def prepare_features(df):
    """Normalize every row. Rows with an unusable palm size are skipped."""
    X_raw = df.drop(columns=["label"])
    y = df["label"].astype(str)

    normalized_rows = []
    valid_indices = []
    for index, row in X_raw.iterrows():
        normalized = normalize_landmarks(row.to_numpy())
        if normalized is not None:
            normalized_rows.append(normalized)
            valid_indices.append(index)

    skipped = len(df) - len(valid_indices)
    if skipped:
        print(f"[WARNING] Skipped {skipped} invalid samples (hand too small/flat).")

    X = pd.DataFrame(normalized_rows, columns=X_raw.columns, index=valid_indices)
    return X, y.loc[valid_indices]


def evaluate_model(model, X, y, title):
    """Print accuracy, macro F1, per-letter report, and a confusion matrix."""
    y_pred = model.predict(X)

    print(f"\n=== {title} ===")
    print(f"Samples tested: {len(y)}")
    print(f"Accuracy: {accuracy_score(y, y_pred) * 100:.2f}%")
    print(
        f"Macro F1: {f1_score(y, y_pred, average='macro', zero_division=0) * 100:.2f}%"
    )

    print("\nPer-letter report (precision / recall / F1):")
    print(classification_report(y, y_pred, zero_division=0))

    labels = sorted(set(y) | set(y_pred))
    matrix = confusion_matrix(y, y_pred, labels=labels)
    print("Confusion matrix (rows = TRUE letter, columns = PREDICTED letter):")
    print(pd.DataFrame(matrix, index=labels, columns=labels))

    wrong = [(t, p) for t, p in zip(y, y_pred) if t != p]
    if wrong:
        print(
            f"\nExample error: true '{wrong[0][0]}' was predicted as '{wrong[0][1]}'."
        )
    else:
        print("\nNo errors on this data set.")


def train_model():
    print("\n--- MODEL TRAINING ---")

    df = read_dataset(DATASET_FILE)
    if df.empty or len(df) < 2:
        print("[ERROR] No/not enough data. Collect data first using Option 1.")
        return

    X, y = prepare_features(df)
    print_label_summary(y)

    counts = y.value_counts()
    if len(counts) < 2:
        print("[ERROR] Need at least 2 different letters to train.")
        return
    if counts.min() < MIN_SAMPLES_PER_LABEL:
        print(
            f"[ERROR] Every letter needs at least {MIN_SAMPLES_PER_LABEL} samples "
            f"(40+ recommended). '{counts.idxmin()}' only has {counts.min()}."
        )
        return

    # --- SEPARATE TRAINING AND TESTING DATA ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"[ERROR] Could not split the data: {e}")
        print("[INFO] Collect more samples per letter and try again.")
        return

    print(
        f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} unseen samples."
    )
    print("Training balanced Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test, "RANDOM 80/20 SPLIT (same recording session)")

    # Re-train on the full dataset before saving for best real-time performance
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"feature_version": MODEL_FEATURE_VERSION, "model": model}, f)

    print(f"\n[SUCCESS] Trained model saved to '{MODEL_FILE}'!")
    print(
        "[NOTE] Scores above come from the 80% model; the saved file is retrained on all data.\n"
    )


def load_model_bundle():
    if not os.path.exists(MODEL_FILE):
        print(
            "[ERROR] No trained model ('model.p') found! Train a model using Option 2 first."
        )
        return None

    with open(MODEL_FILE, "rb") as f:
        model_bundle = pickle.load(f)

    if (
        not isinstance(model_bundle, dict)
        or model_bundle.get("feature_version") != MODEL_FEATURE_VERSION
        or not isinstance(model_bundle.get("model"), RandomForestClassifier)
    ):
        print("[ERROR] Model is outdated. Retrain it using Option 2 first.")
        return None
    return model_bundle["model"]


def evaluate_test_session():
    print("\n--- EVALUATE ON SEPARATE TEST SESSION ---")

    model = load_model_bundle()
    if model is None:
        return

    df = read_dataset(TEST_DATASET_FILE)
    if df.empty:
        print("[ERROR] No test-session data. Use Option 4 to collect some first.")
        return

    X, y = prepare_features(df)
    print_label_summary(y, "Test samples per letter")

    unknown = sorted(set(y) - set(model.classes_))
    if unknown:
        print(
            f"[WARNING] Letters {unknown} are not in the trained model, so they will "
            "always count as errors."
        )

    evaluate_model(model, X, y, "SEPARATE TEST SESSION (never used for training)")
    print()


# ---------------------------------------------------------------------------
# Real-time prediction
# ---------------------------------------------------------------------------
def predict_sign(model, landmarks_input):
    normalized = normalize_landmarks(landmarks_input)
    if normalized is None:
        return None
    model_input = pd.DataFrame([normalized], columns=model.feature_names_in_)
    probabilities = model.predict_proba(model_input)[0]
    ranked_classes = np.argsort(probabilities)

    # --- TEMPORARY DEBUG: top-3 classes ---
    top3 = ranked_classes[::-1][:3]
    print(" | ".join(f"{model.classes_[i]}: {probabilities[i]:.2f}" for i in top3))
    # --------------------------------------

    best_class = int(ranked_classes[-1])
    confidence = probabilities[best_class]
    margin = (
        confidence - probabilities[ranked_classes[-2]]
        if len(ranked_classes) > 1
        else confidence
    )
    if confidence < MIN_PREDICTION_CONFIDENCE or margin < MIN_PREDICTION_MARGIN:
        return None
    return str(model.classes_[best_class])


def realtime_predict():
    print("\n--- REAL-TIME PREDICTION ---")

    model = load_model_bundle()
    if model is None:
        return

    detector = initialize_landmarker()
    cap = cv2.VideoCapture(0)

    current_text = ""
    prediction_history = deque(maxlen=PREDICTION_HISTORY_LENGTH)
    last_output = ""
    release_frames = 0

    print("[INFO] Press 'S' to SPEAK the full constructed text.")
    print("[INFO] Press 'C' to clear the current text.")
    print("[INFO] Press 'Q' to exit.\n")

    cv2.namedWindow("Sign Language Translator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sign Language Translator", 2560, 1440)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam feed.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        hand_landmarks = select_primary_hand(detection_result.hand_landmarks)
        candidate = None
        if hand_landmarks:
            draw_landmarks_on_image(frame, hand_landmarks)
            landmarks_input = []
            base_x = hand_landmarks[0].x
            base_y = hand_landmarks[0].y
            base_z = hand_landmarks[0].z
            for landmark in hand_landmarks:
                landmarks_input.extend(
                    [
                        landmark.x - base_x,
                        landmark.y - base_y,
                        landmark.z - base_z,
                    ]
                )
            candidate = predict_sign(model, landmarks_input)

        prediction_history.append(candidate)
        stable_prediction = None
        if len(prediction_history) == PREDICTION_HISTORY_LENGTH:
            votes = Counter(
                prediction for prediction in prediction_history if prediction
            )
            if votes:
                prediction, vote_count = votes.most_common(1)[0]
                if vote_count >= MIN_STABLE_PREDICTIONS:
                    stable_prediction = prediction

        if stable_prediction:
            release_frames = 0
            if stable_prediction != last_output:
                current_text += stable_prediction
                speak_text(stable_prediction)
                last_output = stable_prediction
        else:
            release_frames += 1
            if release_frames >= PREDICTION_HISTORY_LENGTH:
                last_output = ""

        predicted_text = (
            stable_prediction
            if stable_prediction
            else ("Hold sign steady" if hand_landmarks else "No hands detected")
        )

        cv2.rectangle(frame, (0, 0), (640, 90), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Prediction: {predicted_text}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if stable_prediction else (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Text: {current_text}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "[S] Speak All | [C] Clear | [Q] Quit",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Sign Language Translator", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("s"), ord("S")):
            if current_text:
                speak_text(current_text)
        elif key in (ord("c"), ord("C")):
            current_text = ""
        elif key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


# ---------------------------------------------------------------------------
# Delete / reset
# ---------------------------------------------------------------------------
def delete_data_menu():
    print("\n--- DELETE / RESET DATASET ---")
    print("[1] Training data  (" + DATASET_FILE + ")")
    print("[2] Test-session data  (" + TEST_DATASET_FILE + ")")
    print("[3] Cancel")
    which = input("\nWhich data? (1-3): ").strip()

    if which == "1":
        dataset_file, is_training = DATASET_FILE, True
    elif which == "2":
        dataset_file, is_training = TEST_DATASET_FILE, False
    else:
        return

    if not os.path.exists(dataset_file):
        print("[INFO] No dataset file found.")
        return

    print("[1] Delete a SPECIFIC sign/letter (e.g., delete only 'A')")
    print("[2] Delete EVERYTHING (Full Reset)")
    print("[3] Cancel")

    choice = input("\nSelect an option (1-3): ").strip()

    if choice == "1":
        label_to_delete = (
            input("Enter the letter to delete (e.g., A): ").strip().upper()
        )
        df = read_dataset(dataset_file)

        if label_to_delete in df["label"].astype(str).values:
            df = df[df["label"].astype(str) != label_to_delete]
            df.to_csv(dataset_file, index=False)
            print(f"[SUCCESS] Deleted all entries for letter '{label_to_delete}'.")
            if is_training:
                print("[NOTE] Re-train the model in Option 2 to update the system!")
        else:
            print(f"[INFO] No data found for letter '{label_to_delete}'.")

    elif choice == "2":
        confirm = (
            input("Are you sure you want to DELETE ALL of this data? (Y/N): ")
            .strip()
            .upper()
        )
        if confirm == "Y":
            os.remove(dataset_file)
            if is_training and os.path.exists(MODEL_FILE):
                os.remove(MODEL_FILE)
                print("[SUCCESS] Training data and trained model deleted.")
            else:
                print("[SUCCESS] Data deleted.")


def main():
    while True:
        print("========================================")
        print(" SIGN LANGUAGE TRANSLATOR (Random Forest)")
        print("========================================")
        print("[1] Collect Training Data (landmarks only, append or replace)")
        print("[2] Train Model & Evaluate (80/20 split)")
        print("[3] Run Real-time Prediction (Webcam feed)")
        print("[4] Collect Test-Session Data (separate recording)")
        print("[5] Evaluate on Test-Session Data")
        print("[6] Delete / Reset Dataset")
        print("[7] Exit")

        choice = input("\nSelect an option (1-7): ").strip()

        if choice == "1":
            collect_data(DATASET_FILE, "TRAINING DATA")
        elif choice == "2":
            train_model()
        elif choice == "3":
            realtime_predict()
        elif choice == "4":
            collect_data(TEST_DATASET_FILE, "TEST SESSION")
        elif choice == "5":
            evaluate_test_session()
        elif choice == "6":
            delete_data_menu()
        elif choice == "7":
            print("Exiting application. Goodbye!")
            break
        else:
            print("Invalid selection! Please enter a number between 1 and 7.\n")


if __name__ == "__main__":
    main()
