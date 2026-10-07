CNN - Real-Time Image Classification
📌 Project Overview

This project is a real-time image classification system built using Python, OpenCV, and TensorFlow/Keras.

It uses the pre-trained MobileNetV2 CNN model, trained on the ImageNet dataset, to identify objects captured through the webcam and display the predicted class with its confidence score.

🛠️ Technologies Used
Python
OpenCV
NumPy
TensorFlow
Keras
MobileNetV2
Deep Learning
Convolutional Neural Networks (CNN)
Computer Vision
🧠 How It Works
Loads the pre-trained MobileNetV2 model.
Opens the system webcam using OpenCV.
Captures video frames in real time.
Converts and resizes each frame to 224 × 224.
Preprocesses the image for MobileNetV2.
Classifies the image using the ImageNet-trained model.
Displays the predicted object and confidence percentage on the webcam frame.
Press q to exit.
📂 Project Structure
CNN-Realtime-Image-Classification/
│
├── VedImageClassifier.py
├── README.md
├── requirements.txt
└── venv/

Note: Do not upload the venv/ folder to GitHub. Add it to .gitignore.

🐍 Virtual Environment Setup

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Upgrade pip:

python -m pip install --upgrade pip

Install the required libraries:

pip install -r requirements.txt

To deactivate the virtual environment:

deactivate
▶️ Run the Project

Activate the virtual environment:

venv\Scripts\activate

Run the program:

python VedImageClassifier.py

The webcam will open and display the predicted object with its confidence score.

Press q to close the application.

📦 Requirements

The requirements.txt file should contain the required dependencies:

opencv-python
numpy
tensorflow
👨‍💻 Author

Vedant Dhamal
