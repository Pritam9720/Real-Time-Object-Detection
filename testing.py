# testing.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_labels(label_path="labels.txt"):
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def detect_objects():
    model = load_model("object_detection_model.h5")
    labels = load_labels()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        img = cv2.resize(frame, (64, 64))
        img_array = np.expand_dims(img / 255.0, axis=0)

        # Predict
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        label = labels[class_index]

        # Display
        cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects()
