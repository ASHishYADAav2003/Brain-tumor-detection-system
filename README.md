# Brain-tumor-detection-system
A Brain Tumor Detection System is an intelligent medical imaging system designed to automatically detect and classify brain tumors from MRI (Magnetic Resonance Imaging) scans. It uses image processing and machine learning—especially deep learning techniques—to assist doctors in diagnosing tumors accurately and quickly.


🧠 Brain Tumor Detection System

Deep Learning–based system for detecting and classifying brain tumors from MRI images using Convolutional Neural Networks (CNN).

Overview

This project analyzes brain MRI scans to:

Detect tumor presence (Yes / No)

Classify tumor type (Glioma, Meningioma, Pituitary)

Assist in faster preliminary medical diagnosis

Sample MRI Images
4
Tech Stack

Python

TensorFlow / Keras

OpenCV

NumPy

Project Structure
brain-tumor-detection/
│
├── dataset/
├── models/
├── train.py
├── predict.py
└── README.md

Installation
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt

Usage

Train the model:

python train.py


Run prediction:

python predict.py --image path_to_image.jpg

Model

CNN Architecture

Adam Optimizer

Categorical Crossentropy

Accuracy Metric
