# backend.py

import os
import cv2
import numpy as np
import json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter import messagebox

IMAGE_DIR = "captured_images"
IMAGE_SIZE = (64, 64)
MAX_IMAGES = 1000

def capture_images(class_name):
    save_dir = os.path.join(IMAGE_DIR, class_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "⚠️ Could not access the webcam.")
        return

    count = 0
    messagebox.showinfo("Capture", f"Capturing {MAX_IMAGES} images for '{class_name}'. Hold object steadily.")

    while count < MAX_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break

        img_path = os.path.join(save_dir, f"{class_name}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        frame_display = frame.copy()
        cv2.putText(frame_display, f"{class_name}: {count}/{MAX_IMAGES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Capturing Images", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Capture", f"✅ Captured {count} images for '{class_name}'.")

def train_model():
    try:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

        train_gen = datagen.flow_from_directory(
            IMAGE_DIR,
            target_size=IMAGE_SIZE,
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )

        val_gen = datagen.flow_from_directory(
            IMAGE_DIR,
            target_size=IMAGE_SIZE,
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )

        if train_gen.num_classes < 2:
            messagebox.showerror("Training Error", "At least two classes are required to train the model.")
            return

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(train_gen.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_gen, epochs=10, validation_data=val_gen)

        model.save("object_detection_model.h5")

        with open("labels.json", "w") as f:
            json.dump(train_gen.class_indices, f)

        messagebox.showinfo("Training", "✅ Model trained and saved successfully.")

    except Exception as e:
        messagebox.showerror("Training Error", str(e))

def detect_objects():
    try:
        model = load_model("object_detection_model.h5")

        with open("labels.json", "r") as f:
            labels_dict = {int(v): k for k, v in json.load(f).items()}

    except:
        messagebox.showerror("Error", "⚠️ Train the model first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "⚠️ Could not access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, IMAGE_SIZE)
        img_array = np.expand_dims(img / 255.0, axis=0)

        prediction = model.predict(img_array, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        label = labels_dict.get(class_index, "Unknown")

        if confidence > 0.7:
            display_text = f"{label} ({confidence * 100:.2f}%)"
            color = (0, 255, 0)
        else:
            display_text = "No object detected"
            color = (0, 0, 255)

        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Real-Time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
