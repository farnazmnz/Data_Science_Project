import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import flat_transformer as ft
import linformer as lf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_dir = Path(__file__).resolve().parent.parent / "data" / "features_ravdess"
feature_dir = str(feature_dir)

# Extract and split video groups
video_groups = ft.group_feature_files_by_video(feature_dir, expected_clips=7)
train_videos, val_videos, test_videos = ft.split_videos_no_leakage(video_groups, 0.7, 0.15, seed=42)

train_loader = DataLoader(
    ft.VideoFeatureDataset(video_groups, train_videos),
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    ft.VideoFeatureDataset(video_groups, val_videos),
    batch_size=4,
    shuffle=False
)

test_loader = DataLoader(
    ft.VideoFeatureDataset(video_groups, test_videos),
    batch_size=4,
    shuffle=False
)

# ------------- TRAINING LINFORMER -------------------------- #
model = lf.LinformerTemporalTransformer(
    input_dim=512,
    d_model=256,
    num_classes=8,
    num_clips=7,
    frames_per_clip=12,
    nhead=4,
    num_layers=2,
    ff_dim=512,
    k_proj=16,
    dropout=0.2
).to(device)

# ------------- TRAINING FLAT TRANSFORMER -------------------- #

# Gavin - increased dropout to 0.2 to improve regularization
# model = ft.FlatTemporalTransformer(
#     input_dim=512,
#     d_model=256,
#     num_classes=8,
#     num_clips=7,
#     frames_per_clip=12,
#     nhead=4,
#     num_layers=2,
#     ff_dim=512,
#     dropout=0.2
# ).to(device)

# Gavin - Added label smoothing to help with generalization
criterion = nn.CrossEntropyLoss()
# Gavin - Added weight decay to help with generalization
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

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

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc

best_val_loss = float("inf")
patience = 4
patience_counter = 0
epochs = 25

src_path = Path(__file__).resolve().parent
weights_path = src_path / "transformer_weights" / "best_linformer_weights.pth"

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    print(f"\n===== Epoch {epoch + 1} =====")

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)   # [B, 7, 12, 512]
        labels = labels.to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx == 0:
            print("Features shape:", features.shape)
            print("Outputs shape:", outputs.shape)
            print("Labels:", labels)
            print("Predictions:", preds)
            print("Loss:", loss.item())

    train_loss = total_loss / max(len(train_loader), 1)
    train_acc = 100.0 * correct / max(total, 1)

    val_loss, val_acc = evaluate(model, val_loader, criterion)

    print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")
    print(f"Epoch {epoch + 1} Train Acc : {train_acc:.2f}%")
    print(f"Epoch {epoch + 1} Val Loss : {val_loss:.4f}")
    print(f"Epoch {epoch + 1} Val Acc  : {val_acc:.2f}%")

    # --------------- Early stopping ------------------
    improved = False

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        improved = True

    if improved:
        # Save best model
        torch.save(model.state_dict(), weights_path)
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break

    epoch += 1

# # ------------- Uncomment and run everything below to eval ----------------
src_path = Path(__file__).resolve().parent
# Change last portion of path to current model config
# weights_path = src_path / "transformer_weights" / "best_flat_transformer_weights.pth"
weights_path = src_path / "transformer_weights" / "best_linformer_weights.pth"

model.load_state_dict(torch.load(weights_path))
model.eval()

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

# print("\n===== Flat Transformer Evaluation =====")
print("\n===== Linformer Evaluation =====")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.2f}%")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:")
print(cm)