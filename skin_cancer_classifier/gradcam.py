"""
gradcam.py — Grad-CAM (Gradient-weighted Class Activation Mapping) for model explainability.
Generates heatmap overlays showing which image regions drive predictions.
"""

import os
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B0.
    Hooks into the last convolutional layer to compute gradient-weighted activations.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Default: last convolutional block in EfficientNet backbone
        if target_layer is None:
            # EfficientNet-B0 features[-1] is the last block
            self.target_layer = model.backbone.features[-1]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image tensor.
        
        Args:
            input_tensor: [1, C, H, W] normalized tensor
            target_class: class index to explain; if None, uses predicted class
            
        Returns:
            heatmap: [H, W] numpy array in [0, 1] range
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backpropagate the target class score
        target_score = output[0, target_class]
        target_score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize to input image size
        cam = cv2.resize(cam, (config.IMG_SIZE, config.IMG_SIZE))

        return cam


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap onto the original image."""
    # Convert heatmap to colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Ensure image is uint8 RGB
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Resize image to match heatmap
    image_resized = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))

    # Blend
    overlay = cv2.addWeighted(image_resized, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def visualize_gradcam_samples(
    model: nn.Module,
    image_paths: List[str],
    labels: List[int],
    transform,
    samples_per_class: int = 2,
) -> None:
    """Generate and save Grad-CAM visualizations for sample images from each class."""
    from albumentations.pytorch import ToTensorV2
    import albumentations as A

    device = config.DEVICE
    grad_cam = GradCAM(model)

    # Group paths by class
    class_paths = {i: [] for i in range(config.NUM_CLASSES)}
    for path, label in zip(image_paths, labels):
        class_paths[label].append(path)

    rng = np.random.RandomState(config.SEED)
    n_classes = config.NUM_CLASSES
    fig, axes = plt.subplots(n_classes, samples_per_class * 2,
                              figsize=(samples_per_class * 6, n_classes * 3))

    # Simple transform for Grad-CAM input (no augmentation)
    cam_transform = A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])

    for i in range(n_classes):
        class_name = config.IDX_TO_CLASS[i]
        paths = class_paths[i]
        if len(paths) == 0:
            continue
        chosen = rng.choice(paths, min(samples_per_class, len(paths)), replace=False)

        for j, img_path in enumerate(chosen):
            # Read original image
            orig_img = cv2.imread(img_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            display_img = cv2.resize(orig_img, (config.IMG_SIZE, config.IMG_SIZE))

            # Prepare tensor
            transformed = cam_transform(image=orig_img)
            input_tensor = transformed["image"].unsqueeze(0).to(device)

            # Generate Grad-CAM
            heatmap = grad_cam.generate(input_tensor, target_class=i)
            overlay = overlay_heatmap(display_img, heatmap)

            # Original image
            col_orig = j * 2
            col_cam = j * 2 + 1
            axes[i][col_orig].imshow(display_img)
            axes[i][col_orig].set_title(f"Original", fontsize=8)
            axes[i][col_orig].axis("off")

            # Grad-CAM overlay
            axes[i][col_cam].imshow(overlay)
            axes[i][col_cam].set_title(f"Grad-CAM", fontsize=8)
            axes[i][col_cam].axis("off")

        # Label
        axes[i][0].set_ylabel(class_name, fontsize=7, rotation=0,
                               labelpad=80, va="center")

    fig.suptitle("Grad-CAM Explainability — Per Class", fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(config.GRADCAM_DIR, "gradcam_overview.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[GRAD-CAM] Visualization saved → {save_path}")

def generate_gradcam_for_inference(
    model: nn.Module,
    image_path: str,
    output_path: str,
    device: torch.device,
    target_class: Optional[int] = None,
) -> None:
    """Generate and save Grad-CAM visualization for a single image during inference."""
    from albumentations.pytorch import ToTensorV2
    import albumentations as A

    grad_cam = GradCAM(model)

    cam_transform = A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])

    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Could not read image at {image_path}")
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    display_img = cv2.resize(orig_img, (config.IMG_SIZE, config.IMG_SIZE))

    transformed = cam_transform(image=orig_img)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    heatmap = grad_cam.generate(input_tensor, target_class=target_class)
    overlay = overlay_heatmap(display_img, heatmap)

    # Save
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(display_img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM (Class: {config.IDX_TO_CLASS[target_class] if target_class is not None else 'Predicted'})")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
