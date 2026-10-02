import mediapipe as mp
import cv2
import time
import os

from create_dataset import CreateDataset

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 30

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)


def main():
    for i in range(number_of_classes):
        if not os.path.exists(os.path.join(DATA_DIR, str(i))):
            os.makedirs(os.path.join(DATA_DIR, str(i)))

        print(f"Collecting data for Class {i}")

        done = False

        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            while True:
                attempt = 0
                success, img = capture.read()

                while not success and attempt < 5:
                    time.sleep(0.2)
                    success, img = capture.read()
                    attempt += 1
                if not success:
                    print("Failed to read frame")
                    break

                img = cv2.flip(img, 1)
                height, width, _ = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            img,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                        finger_tip = {
                            "Thumb": hand_landmarks.landmark[4],
                            "Index": hand_landmarks.landmark[8],
                            "Middle": hand_landmarks.landmark[12],
                            "Ring": hand_landmarks.landmark[16],
                            "Pinky": hand_landmarks.landmark[20],
                        }

                        for name, landmark in finger_tip.items():
                            x, y = int(landmark.x * width), int(landmark.y * height)
                            cv2.putText(
                                img,
                                name,
                                (x, y - 10),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.0,
                                (255, 255, 255),
                                1,
                            )

                            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    img,
                    "If Ready, Press Q!",
                    (250, 80),
                    cv2.FONT_HERSHEY_PLAIN,
                    5.0,
                    (102, 255, 0),
                    2,
                )
                cv2.imshow("Fundamentals Project", img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        counter = 0
        while counter < dataset_size:
            success, img = capture.read()
            img = cv2.flip(img, 1)
            cv2.imshow("Fundamentals Project", img)
            cv2.waitKey(1)
            cv2.imwrite(os.path.join(DATA_DIR, str(i), f"{counter}.jpg"), img)
            counter += 1

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    CreateDataset()
