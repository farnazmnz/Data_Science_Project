# Spatiotemporal Facial Emotion Recognition
Hybrid CNN–Transformer and CNN–LSTM Models using RAVDESS Dataset

## Overview
This project explores spatiotemporal facial emotion recognition from video sequences using hybrid deep learning architectures. We compare CNN-LSTM, CNN-Transformer, and CNN-Linformer models.

A pretrained ResNet18 is used to extract spatial features, while temporal dependencies are modeled using Transformer and LSTM-based architectures.

---

## Project Goals
- Extract spatial features from facial video frames  
- Model temporal relationships between frames  
- Compare Transformer vs LSTM performance  
- Evaluate models using accuracy and F1-score  

---

## Dataset
We use the **[RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

Emotion Classes:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprise

---
## Data Preprocessing

Before training, each video is converted into short clips.

Steps:
1. Load RAVDESS video sequences
2. Detect face using MediaPipe face landmarks
3. Crop facial bounding box
4. Resize frames to 128×128
5. Normalize using ImageNet mean/std
6. Split videos into **12-frame clips**
7. Store clips for model input

This clip-based approach reduces computational cost while preserving temporal information.

---
## Models
### CNN + LSTM
Temporal modeling using LSTM baseline.

### CNN + Transformer
Self-attention based temporal modeling.

### CNN + Linformer
Efficient transformer variant.

---

## Results

| Model | Accuracy | Macro F1 |
|------|---------|---------|
| CNN-LSTM | 89.42% | 0.8914 |
| CNN-Transformer | **94.97%** | **0.9482** |
| CNN-Linformer | 89.68% | 0.8875 |

