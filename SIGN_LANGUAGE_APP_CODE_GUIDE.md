# Sign Language App: Code Guide

This guide describes the current implementation in [`sign_language_app.py`](./sign_language_app.py). It is intended to help you understand the program and explain it during a project presentation.

## 1. What the program does

The app classifies a hand pose from webcam landmarks and outputs one label at a time. Its workflow is:

1. **Collect labeled examples:** MediaPipe detects a hand and the program records its 21 three-dimensional landmarks in a CSV file with a label.
2. **Train and evaluate:** A Random Forest classifier learns from the training CSV, reports holdout metrics, then is retrained on all valid training examples and saved.
3. **Predict from the webcam:** The saved classifier predicts a label for each frame. Confidence and recent-frame voting filters reduce unstable predictions before a label is displayed, appended to the current text, and spoken.
4. **Optionally evaluate a separate session:** A second CSV can be collected and evaluated against the saved model without using those examples for training.

This is a static-pose classifier. It does not recognize motion-based signs, facial expressions, or complete signed sentences. The text it builds is a sequence of recognized labels.

## 2. Running the program

From the project directory, run:

```bash
python sign_language_app.py
```

On macOS or Linux, if the project virtual environment is named `venv`, you can run:

```bash
./venv/bin/python sign_language_app.py
```

The `if __name__ == "__main__":` guard at the end of the file calls `main()` when the script is run directly. The program then displays a seven-option menu.

The project [`readme.md`](./readme.md) specifies Python 3.12.9.

The app uses a webcam and OpenCV display windows for collection and real-time prediction. If `hand_landmarker.task` is missing from the current working directory, the program attempts to download it from Google's MediaPipe model storage when it initializes the detector; the first run therefore needs network access.

### Dependencies

The script imports OpenCV (`cv2`), MediaPipe, NumPy, Pandas, `pyttsx3`, and scikit-learn. The current [`requirements.txt`](./requirements.txt) lists OpenCV, MediaPipe, and scikit-learn, but does not list NumPy, Pandas, or `pyttsx3`, which the script also imports. A fresh environment needs all of these packages installed for the script to start.

## 3. Files and settings

The filenames are plain relative paths, so they are resolved from the process's **current working directory**:

| Setting | Current value | Purpose |
| --- | --- | --- |
| `DATASET_FILE` | `hand_landmarks_data.csv` | Labeled training examples |
| `TEST_DATASET_FILE` | `test_session_data.csv` | Separate evaluation-session examples |
| `MODEL_FILE` | `model.p` | Pickled model bundle |
| `TASK_FILE` | `hand_landmarker.task` | MediaPipe hand-landmarker asset |
| `MODEL_FEATURE_VERSION` | `2` | Prevents loading a model saved for a different feature format |
| `MIN_PREDICTION_CONFIDENCE` | `0.7` | Minimum accepted top-class probability |
| `MIN_PREDICTION_MARGIN` | `0.1` | Minimum gap between the top two class probabilities |
| `PREDICTION_HISTORY_LENGTH` | `8` | Number of recent frames used for stability voting |
| `MIN_STABLE_PREDICTIONS` | `6` | Minimum matching votes required from the eight frames |
| `MIN_SAMPLES_PER_LABEL` | `5` | Minimum valid training samples per label; 40 or more is recommended in the code |

The hand-landmarker is configured for at most one hand and a minimum detection confidence of `0.5`. The code does not save webcam photos.

## 4. Hand detection and feature preparation

### MediaPipe and camera frames

`initialize_landmarker()` ensures the task asset exists and creates a MediaPipe Tasks `HandLandmarker`. During collection and prediction, the app reads from camera index `0`, mirrors each frame for display, converts the image from OpenCV's BGR format to RGB, and passes it to MediaPipe.

`draw_landmarks_on_image()` draws lines between the hand's 21 landmark points and draws a circle at each point. The drawing is only an on-screen overlay; it is not written to disk.

`select_primary_hand()` returns `None` if no hands are supplied. Although it can rank multiple hands, the detector is currently configured with `num_hands=1`, so normally there is at most one hand to select. If called with multiple hands, its score uses a signed linear expression based on the bounding-box center, then negative box area as a tie-breaker; this is not a geometric distance calculation.

### Landmark normalization

Each detected hand has 21 landmarks, each with `x`, `y`, and `z` values, for 63 numeric features.

`normalize_landmarks()`:

1. Requires exactly 63 values.
2. Reshapes them into 21 `(x, y, z)` points.
3. Subtracts the wrist (landmark `0`) from every point, making the wrist the origin.
4. Divides all coordinates by the 3D distance from the wrist to landmark `9` (near the base of the middle finger).
5. Returns the resulting 63 values as a flat list.

If the wrist-to-landmark-9 scale is nearly zero, it returns `None` because the sample cannot be normalized safely. Training skips those rows. A wrong feature count raises a `ValueError`.

## 5. Dataset format and collection

The CSV header contains `x0, y0, z0` through `x20, y20, z20`, followed by `label`. Each collected row therefore contains 63 wrist-relative coordinates and one label. Normalization is applied later when preparing data for training or evaluation.

`read_dataset()` returns an empty DataFrame if the file is missing or empty. `save_samples()` supports two modes:

- **Append:** Keep all existing rows and add the new samples.
- **Replace:** Remove existing rows for the selected label and add the new samples, leaving other labels untouched.

When a label already has saved samples, the user can append, replace, or cancel. New rows are held in memory during collection; the CSV is only updated after the camera loop ends and at least one row has been collected. If collection is cancelled before any hand samples are captured, the existing data is unchanged.

### Collection controls

Menu options **1** and **4** both call `collect_data()`. The program asks for a label and a sample target (default `200`), opens the camera, and waits for **S** in the camera window before recording. It captures a row whenever a hand is detected while collection is active. **Q** ends collection and saves any rows already collected; reaching the requested target also ends the loop. No photos are saved.

- **Option 1** writes to `hand_landmarks_data.csv` for training.
- **Option 4** writes to `test_session_data.csv` for separate evaluation.

## 6. Training and model evaluation

`train_model()` reads the training CSV and prepares the features:

- The `label` column becomes the target values (`y`); the other columns are the input features (`X`).
- Every row is normalized with `normalize_landmarks()`.
- Rows with an unusable palm scale are skipped and reported.
- At least two labels are required, and every label needs at least five valid samples.

The data is split into training and holdout sets using an 80/20 split with `random_state=42` and `stratify=y`. If scikit-learn cannot make the split, the program prints the error and asks for more samples.

The classifier is a `RandomForestClassifier` configured with:

- `n_estimators=100`
- `min_samples_leaf=2`
- `class_weight="balanced_subsample"`
- `random_state=42`
- `n_jobs=-1`

The first fit uses the 80% training portion. The remaining 20% is used for evaluation. `evaluate_model()` prints accuracy, macro F1, a per-label classification report, a confusion matrix, and the first example error if there is one. These metrics describe a random split from the same recorded dataset; they do not establish performance on a new signer or recording setup.

After evaluation, the classifier is fitted again using all valid training data and saved in `model.p` as a dictionary containing the feature version and the model.

## 7. Separate-session evaluation

Option **5** loads the saved model and evaluates it against `test_session_data.csv`. The test-session CSV is not used during training. If it contains labels that the model was not trained to recognize, the program prints a warning; those labels will be counted as incorrect predictions.

The shared `evaluate_model()` function reports the same metrics as training evaluation. For a meaningful separate-session result, collect those examples independently from the training examples.

## 8. Loading and using the saved model

`load_model_bundle()` checks for `model.p`, loads it with `pickle`, and verifies that:

- The saved value is a dictionary.
- Its `feature_version` matches `MODEL_FEATURE_VERSION`.
- Its `model` is a `RandomForestClassifier`.

If the model file is missing or fails these checks, the program prints a message and returns without starting prediction or test evaluation. Since pickle files can execute code when loaded, only use a model file from a trusted source.

`predict_sign()` normalizes one hand's 63 coordinates and asks the classifier for probabilities. It rejects a prediction if the top probability is below `0.7` or if the gap between the top and second-highest probabilities is below `0.1`. Rejected predictions are represented by `None`.

## 9. Real-time prediction and speech

Option **3** loads the model and starts webcam prediction. It uses the same wrist-relative coordinates and normalization as training.

For each frame, the program puts the candidate prediction (or `None`) into an eight-frame history. It accepts a stable label when at least six of the non-`None` predictions in that history agree. The program then:

- Displays the stable label, or displays `Hold sign steady` / `No hands detected`.
- Appends a stable label to the on-screen text only when it differs from the last output, so holding one pose does not add it continuously.
- Speaks each newly appended label using `pyttsx3` in a background daemon thread.
- Allows the same label to be appended again after eight frames without a stable prediction reset the last output.

Keyboard controls in the prediction window:

- **S:** Speak the complete text assembled so far.
- **C:** Clear the assembled text.
- **Q:** Exit prediction mode.

Speech errors are printed as `Speech error: ...`; they are handled inside the speech worker so they do not terminate the camera loop.

## 10. Deleting data

Option **6** lets the user choose training data or test-session data, then delete one label, delete the selected dataset entirely, or cancel.

- Deleting one training label rewrites the training CSV without that label and reminds the user to retrain. It does **not** automatically update or delete `model.p`.
- Deleting all training data also deletes `model.p` if it exists.
- Deleting test-session data does not affect the trained model.
- Deletion does not remove `hand_landmarker.task`.

## 11. Main menu

`main()` repeatedly displays these options until the user selects **7**:

1. Collect training data.
2. Train the model and evaluate its 80/20 split.
3. Run real-time webcam prediction.
4. Collect separate test-session data.
5. Evaluate the saved model on test-session data.
6. Delete or reset a dataset.
7. Exit.

An input outside `1`–`7` displays an invalid-selection message.

## 12. End-to-end data flow

1. OpenCV reads and mirrors a webcam frame.
2. MediaPipe detects up to one hand and returns 21 `(x, y, z)` landmarks.
3. The wrist is used as the coordinate origin; the points are scaled by the wrist-to-landmark-9 distance.
4. Collection stores the 63 wrist-relative coordinates and label in the selected CSV. No images are stored.
5. Training normalizes valid rows, fits and evaluates a Random Forest on a stratified 80/20 split, then refits on all valid training rows and saves the versioned model bundle.
6. Prediction applies matching normalization, confidence and margin filters, then eight-frame voting before displaying and speaking a label.

## 13. Presentation summary and limitations

One concise description of the project is:

> “Our app uses MediaPipe to locate 21 points on a hand. We save wrist-relative landmark coordinates as labeled examples, normalize them to reduce sensitivity to hand position and scale, and train a Random Forest classifier. During webcam use, confidence checks and recent-frame voting filter predictions before the app displays and speaks each recognized label.”

Important limitations:

- The classifier recognizes individual static hand poses, not motion-based signs or whole signed sentences.
- The model learns only the labels and variations represented in its training CSV.
- The reported 80/20 metrics use a random holdout from the training data and may not reflect results for new people, lighting, cameras, or environments.
- Prediction filters improve stability but cannot guarantee a correct classification.
- Webcam availability, a working OpenCV display, the required Python packages, and access to the MediaPipe task file on first run are needed for the corresponding features.
