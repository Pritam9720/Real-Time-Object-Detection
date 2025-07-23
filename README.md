# Real-Time Object Detection using CNN, OpenCV and TensorFlow

## 👨‍💻 Developed by
**Pritam Patil** 

## 🧠 Project Overview
This project focuses on real-time object detection using pre-trained CNN models and OpenCV. It captures images using webcam, trains a CNN model using TensorFlow, and predicts live objects via a GUI interface.

## 🔧 Technologies Used
- Python 3.x  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Tkinter (for GUI)  
- ImageDataGenerator  

## 🧱 Project Architecture
1. 📸 Auto-capture images using webcam  
2. 🧠 Train CNN model on custom dataset  
3. 🎯 Real-time detection using webcam feed  
4. 📊 Display prediction with confidence %  

## 📂 Folder Structure
/captured_images/
└── Classwise folders with training images
object_detection_model.h5 ← Saved CNN model
labels.txt ← Class labels
main.py ← GUI + Detection logic
README.md ← Project documentation

Then use the GUI to:

Enter class name

Auto-capture 1000 images

Train model

Test real-time detection

✅ Features
Real-time object detection with >70% confidence threshold
GUI using Tkinter
CNN trained via ImageDataGenerator
Save model and class labels

🔮 Future Scope
Integrate object tracking (SORT/Deep SORT)
Enable alert systems (SMS, alarm, etc.)
Add custom object datasets
Enhance performance on low-end devices

📚 References
OpenCV Docs
YOLO Paper
TensorFlow Detection Models
PyImageSearch

