import os
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Setup paths
base_dir = Path("c:/Users/Asus/OneDrive/Documents/GitHub/Project/skin_cancer_v3")
train_dir = base_dir / "data" / "Train"
figures_dir = base_dir / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# 1. Class Distribution
classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
class_counts = {}
for c in classes:
    files = list((train_dir / c).glob('*.jpg'))
    # Also check other extensions just in case
    if len(files) == 0:
        files = list((train_dir / c).glob('*.*'))
    class_counts[c] = len(files)

# Save distribution plot
plt.figure(figsize=(12, 6))
sns.barplot(x=list(class_counts.values()), y=list(class_counts.keys()), palette="viridis")
plt.title("Class Distribution in Training Dataset")
plt.xlabel("Number of Images")
plt.ylabel("Class")
plt.tight_layout()
plt.savefig(figures_dir / "class_distribution.png")
plt.close()

# 2. Basic Dimension Analysis & Sample
# We'll take max 50 images per class to quickly gauge dimensions
dimensions = []
samples = {}

for c in classes:
    files = list((train_dir / c).glob('*.jpg'))
    if len(files) == 0:
        files = list((train_dir / c).glob('*.*'))
    
    if len(files) > 0:
        samples[c] = str(files[0])
        
    for f in files[:50]:
        img = cv2.imread(str(f))
        if img is not None:
            dimensions.append({
                'Class': c,
                'Width': img.shape[1],
                'Height': img.shape[0]
            })

if dimensions:
    df_dims = pd.DataFrame(dimensions)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_dims, x='Width', y='Height', hue='Class')
    plt.title("Image Dimensions Sample")
    plt.tight_layout()
    plt.savefig(figures_dir / "image_dimensions.png")
    plt.close()

# Save samples plot
if samples:
    plt.figure(figsize=(15, 10))
    for i, (c, path) in enumerate(samples.items()):
        plt.subplot(3, 3, i+1)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(c)
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(figures_dir / "sample_images.png")
    plt.close()

# Save stats to JSON
stats = {
    "total_images": sum(class_counts.values()),
    "class_counts": class_counts,
    "unique_widths": list(set([d['Width'] for d in dimensions])) if dimensions else [],
    "unique_heights": list(set([d['Height'] for d in dimensions])) if dimensions else []
}

with open(base_dir / "reports" / "eda_stats.json", "w") as f:
    json.dump(stats, f, indent=4)

print("EDA completed successfully. Stats saved to eda_stats.json")
