import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# Resolve paths relative to the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))

# Constants
INTRINSICS = (2123, 2123, 1500, 2000)
CLASSES = ['back', 'bottom', 'front', 'side', 'top']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def get_bbox_from_txt(txt_path, img_w, img_h):
    """
    Parses YOLO segmentation labels to extract bounding box.
    """
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    line = lines[0].strip().split()
    coords = [float(x) for x in line[1:]]
    xs = coords[0::2]
    ys = coords[1::2]
    x1, x2 = min(xs) * img_w, max(xs) * img_w
    y1, y2 = min(ys) * img_h, max(ys) * img_h
    return int(x1), int(x2), int(y1), int(y2)

def crop_mango(img_rgb, filename, yolo_model):
    """
    Crops the mango from the image, trying the annotation txt first, then YOLO segmenter fallback.
    """
    h, w = img_rgb.shape[:2]
    
    # Try text file first
    txt_filename = os.path.splitext(filename)[0] + '.txt'
    txt_path = os.path.join(WORKSPACE_DIR, 'labels/train', txt_filename)
    bbox = get_bbox_from_txt(txt_path, w, h)
    
    if bbox is not None:
        x1, x2, y1, y2 = bbox
        # Pad slightly to capture the entire fruit context
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.05)
        x1 = max(0, x1 - pad_x)
        x2 = min(w, x2 + pad_x)
        y1 = max(0, y1 - pad_y)
        y2 = min(h, y2 + pad_y)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size > 0:
            return crop

    # Fallback to YOLO model
    results = yolo_model(img_rgb, verbose=False)[0]
    if len(results.boxes) > 0:
        box = results.boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size > 0:
            return crop

    # Final fallback: return the original image
    return img_rgb

class MangoImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_crop = row['crop']
        label = row['label']
        
        # Convert to PIL Image
        img_pil = Image.fromarray(img_crop)
        
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)
            
        return img_tensor, label

def main():
    print("Loading datasets...")
    df_y = pd.read_csv(os.path.join(WORKSPACE_DIR, 'labels/train/mango_y_features.csv'))
    
    print("Loading YOLO model for fallback cropping...")
    yolo_model = YOLO(os.path.join(WORKSPACE_DIR, "models/YoloV11n/runs/segment/mango_seg_v1-4/weights/best.pt"))
    
    print("Pre-processing and cropping images (this may take up to a minute)...")
    crops = []
    labels = []
    filenames = []
    mango_ids = []
    
    for idx, row in df_y.iterrows():
        fn = row['filename']
        orient = row['orientation']
        mango_id = row['mango_id']
        
        img_path = os.path.join(WORKSPACE_DIR, 'images/train', fn)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crop = crop_mango(img_rgb, fn, yolo_model)
        crop_resized = cv2.resize(crop, (224, 224))
        
        crops.append(crop_resized)
        labels.append(CLASS_TO_IDX[orient])
        filenames.append(fn)
        mango_ids.append(mango_id)
        
    df_dataset = pd.DataFrame({
        'filename': filenames,
        'crop': crops,
        'label': labels,
        'mango_id': mango_ids
    })
    print(f"Preprocessed {len(df_dataset)} images.")

    # Split dataset using GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_dataset, df_dataset['label'], groups=df_dataset['mango_id']))
    
    df_train = df_dataset.iloc[train_idx].reset_index(drop=True)
    df_test = df_dataset.iloc[test_idx].reset_index(drop=True)
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MangoImageDataset(df_train, transform=train_transform)
    test_dataset = MangoImageDataset(df_test, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize MobileNetV3-Small model
    print("\nInitializing pre-trained MobileNetV3-Small...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Fine-tune classifier head
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, len(CLASSES))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # Training Loop
    epochs = 15
    print(f"Training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2%}")

    # Evaluation
    print("\nEvaluating model on unseen test subjects...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    print(f"CNN Test Accuracy: {acc:.2%}")
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=CLASSES))

    # Save Confusion Matrix
    save_dir = SCRIPT_DIR
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Oranges')
    plt.title("CNN Orientation Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = os.path.join(save_dir, "orientation_confusion_matrix_cnn.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # Train final model on ALL data
    print("\nTraining final model on the entire dataset...")
    full_dataset = MangoImageDataset(df_dataset, transform=train_transform)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    final_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    final_model.classifier[3] = nn.Linear(num_features, len(CLASSES))
    final_model = final_model.to(device)
    
    optimizer = optim.AdamW(final_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    for epoch in range(epochs):
        final_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in full_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / len(full_dataset)
        epoch_acc = correct / total
        print(f"Final Epoch {epoch+1:02d}/{epochs:02d} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2%}")

    # Save CNN weights
    model_path = os.path.join(save_dir, "orientation_cnn.pth")
    torch.save(final_model.state_dict(), model_path)
    print(f"Saved CNN weights to: {model_path}")
    
    # Save class mapping
    class_mapping_path = os.path.join(save_dir, "class_mapping.json")
    with open(class_mapping_path, 'w') as f:
        json.dump(CLASSES, f)
    print(f"Saved class mapping configuration to: {class_mapping_path}")

if __name__ == '__main__':
    main()
