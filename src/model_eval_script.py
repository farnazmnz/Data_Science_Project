import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import flat_transformer as ft
import linformer as lf
import lstm as lstm
from sklearn.metrics import confusion_matrix, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_dir = Path(__file__).resolve().parent.parent / "data" / "features_ravdess"
feature_dir = str(feature_dir)

# Extract and split video groups
video_groups = ft.group_feature_files_by_video(feature_dir, expected_clips=7)
train_videos, val_videos, test_videos = ft.split_videos_no_leakage(video_groups, 0.7, 0.15, seed=42)

test_loader = DataLoader(
    ft.VideoFeatureDataset(video_groups, test_videos),
    batch_size=4,
    shuffle=False
)

# -------------- LSTM SETUP --------------------------- #
# model = lstm.CNN_LSTM_Model(
#     input_dim=512,
#     hidden_dim=256,
#     num_layers=2,
#     num_classes=8,
#     dropout=0.2
# ).to(device)

# ------------- LINFORMER SETUP-------------------------- #
# model = lf.LinformerTemporalTransformer(
#     input_dim=512,
#     d_model=256,
#     num_classes=8,
#     num_clips=7,
#     frames_per_clip=12,
#     nhead=4,
#     num_layers=2,
#     ff_dim=512,
#     k_proj=16,
#     dropout=0.2
# ).to(device)

# ------------- FLAT TRANSFORMER SETUP-------------------- #
model = ft.FlatTemporalTransformer(
    input_dim=512,
    d_model=256,
    num_classes=8,
    num_clips=7,
    frames_per_clip=12,
    nhead=4,
    num_layers=2,
    ff_dim=512,
    dropout=0.2
).to(device)

criterion = nn.CrossEntropyLoss()

src_path = Path(__file__).resolve().parent
# Change last portion of path to current model config
weights_path = src_path / "transformer_weights" / "best_flat_transformer_weights.pth"
# weights_path = src_path / "transformer_weights" / "best_linformer_weights.pth"
# weights_path = src_path / "transformer_weights" / "best_lstm_weights.pth"


model.load_state_dict(torch.load(weights_path))

def evaluate_test(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # store for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc, all_preds, all_labels

test_loss, test_acc, all_preds, all_labels = evaluate_test(
    model,
    test_loader,
    criterion
)

print("\n===== Flat Transformer Evaluation =====")
# print("\n===== LSTM Evaluation =====")
# print("\n===== Linformer Evaluation =====")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# Treats all classes equally
f1_macro = f1_score(all_labels, all_preds, average="macro")
# Per class f1
f1_per_class = f1_score(all_labels, all_preds, average=None)

class_map = {0:"neutral", 1:"calm", 2:"happy", 3:"sad", 4:"angry", 5:"fearful", 6:"disgust", 7:"surprise"}

print(f"\nMacro f1 score: {f1_macro:.4f}")
print("\nPer-class f1:")
for i, f1 in enumerate(f1_per_class):
    print(f"class {class_map.get(i)}: {f1:.4f}")