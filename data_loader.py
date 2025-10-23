import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from collections import Counter

# File paths (replace with your paths)
hyperspectral_file_path = '/content/PaviaU.mat'
ground_truth_file_path = '/content/PaviaU_gt.mat'

# Load the data
hyperspectral_data = loadmat(hyperspectral_file_path)
hyperspectral_image = hyperspectral_data['paviaU']

ground_truth_data = loadmat(ground_truth_file_path)
ground_truth_image = ground_truth_data['paviaU_gt']

# Shapes
rows, cols, bands = hyperspectral_image.shape
print(f"\nHyperspectral Image Shape: {hyperspectral_image.shape} (Rows x Cols x Bands)")
print(f"Ground Truth Shape        : {ground_truth_image.shape}")

# Total pixels
total_pixels = rows * cols
print(f"Total Pixels              : {total_pixels:,}")
print(f"Spectral Bands            : {bands}")

# Mapping class labels to names
class_names = {
    1: "Asphalt",
    2: "Meadows",
    3: "Gravel",
    4: "Trees",
    5: "Painted metal sheets",
    6: "Bare Soil",
    7: "Bitumen",
    8: "Self-Blocking Bricks",
    9: "Shadows"
}

# Count pixels per class
class_counts = Counter(ground_truth_image.flatten())
print("\nPixel Count & Percentage per Class:")
for cls, count in sorted(class_counts.items()):
    name = class_names.get(cls, "Background" if cls == 0 else "Unknown")
    percentage = (count / total_pixels) * 100
    print(f"Class {cls:<2} ({name:<22}): {count:>7,} pixels ({percentage:5.2f}%)")

# Number of classes (excluding background)
num_classes = len([c for c in class_counts if c != 0])
print(f"\nNumber of Classes (excluding background): {num_classes}")

# Background pixel percentage
background_pixels = class_counts.get(0, 0)
background_percentage = (background_pixels / total_pixels) * 100
print(f"Background Pixels                       : {background_pixels:,} ({background_percentage:.2f}%)")

# Most and least common class (excluding background)
filtered_counts = {k: v for k, v in class_counts.items() if k != 0}
most_common = max(filtered_counts.items(), key=lambda x: x[1])
least_common = min(filtered_counts.items(), key=lambda x: x[1])
print(f"\nMost Common Class  : {most_common[0]} ({class_names[most_common[0]]}) - {most_common[1]:,} pixels")
print(f"Least Common Class : {least_common[0]} ({class_names[least_common[0]]}) - {least_common[1]:,} pixels")

# Visualize ground truth
plt.figure(figsize=(6, 6))
plt.imshow(ground_truth_image, cmap='jet')
plt.title('Ground Truth Labels')
plt.colorbar()
plt.axis('off')
plt.show()

# Bar chart of class distribution
plt.figure(figsize=(10, 6))
classes = list(class_counts.keys())
counts = list(class_counts.values())
labels = [class_names.get(cls, "Background") if cls != 0 else "Background" for cls in classes]

plt.bar(labels, counts, color='mediumseagreen')
plt.xlabel("Land Cover Class")
plt.ylabel("Number of Pixels")
plt.title("Pixel Count per Land Cover Class")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Visualize selected spectral bands
band_20 = hyperspectral_image[:, :, 20]
band_60 = hyperspectral_image[:, :, 60]
band_100 = hyperspectral_image[:, :, 100]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(band_20, cmap='gray')
ax[0].set_title('Band 20')
ax[0].axis('off')

ax[1].imshow(band_60, cmap='gray')
ax[1].set_title('Band 60')
ax[1].axis('off')

ax[2].imshow(band_100, cmap='gray')
ax[2].set_title('Band 100')
ax[2].axis('off')

plt.tight_layout()
plt.show()


import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, precision_score, recall_score, f1_score
import seaborn as sns
import torch.nn as nn
import time
from einops import rearrange
import torch.cuda as cuda
from thop import profile

def load_pavia_university(image_file, gt_file):
    print("Loading Pavia University dataset...")
    image_data = scipy.io.loadmat(image_file)['paviaU']
    ground_truth = scipy.io.loadmat(gt_file)['paviaU_gt']
    print(f"Image data shape: {image_data.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    return image_data, ground_truth