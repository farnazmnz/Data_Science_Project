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
- Analyze generalization across datasets  

---

## Dataset
We use the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

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

