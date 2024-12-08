# AIRA - Machine Learning

<p align="center" ><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/512px-TensorFlow_logo.svg.png?20211220215155" width="325"/> </p>

## Overview

Machine Learning (ML) is the core of AIRA, responsible for analyzing and evaluating piano performance based on user-uploaded audio. Our ML models are designed to recognize notes, rhythm accuracy, and chord structures, providing users with real-time and detailed feedback.

## Feature
### 🎯 Key ML Functionalities:
- **Audio Preprocessing**: Extract features such as pitch, tempo, and chroma using Librosa.
- **Chord Recognition**: Detect and match chords with ideal references using a deep learning model.
- **Real-Time Feedback**: Analyze user performance and provide actionable improvement suggestions.
- **Progress Tracking**: Monitor and evaluate users’ skill improvements over time.

## Technology Stack
### Libraries and Frameworks
| Library        | Functionality              |
|:------------------:|:--------------------------|
| TensorFlow         | Build and train neural network models           | 
| Keras              | High-level API for neural network design        | 
| Librosa            | Audio feature extraction (pitch, chroma, tempo) | 
| NumPy              | Numerical computations                          | 
| Pandas             | Dataset manipulation and analysis               | 

## Dataset
- **Kaggle Piano Dataset**: Primary dataset for training the model.
- **Magenta Dataset**: Secondary dataset for chord variety.

### Data Preprocessing Steps:
1. Clean audio data to remove noise.
2. Standardize audio file lengths.
3. Extract features using Librosa (pitch, chroma, tempo).
4. Save processed data in CSV format for training.

## Model Architecture
Our model employs a **Deep Neural Network** (DNN) for note and chord detection:
- **Input**: Extracted audio features (MFCC, chroma, tempo).
- **Hidden** Layers: Fully connected layers with ReLU activation.
- **Output**: Note and chord predictions using softmax for classification.
### Training Details:
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Evaluation Metrics**: Accuracy, Precision, Recall

## Performance Metrics
| Metric              | Value              |
|:------------------  |:--------------------------|
| Training Accuracy   | 92%        | 
| Validation Accuracy | 89%        | 
| Precision           | 88%        | 
| Recal               | 87%        | 

## Getting Started 
### ML Environtment Setup Guide:
🚀 Set up your machine learning environment
[Click here for machine learning Setup Guide](https://github.com/TCHWG/)




