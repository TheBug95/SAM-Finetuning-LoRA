"""
Example script showing advanced usage of the SAM LoRA training framework.
This demonstrates:
- Custom data loading
- Custom loss functions
- Custom callbacks
- Model ensembling
"""
import sys
from pathlib import Path

# Add Core Modules to path
sys.path.insert(0, str(Path(__file__).parent / "Core Modules"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from config import get_default_config
from dataset import COCOSegmentationDataset, get_transforms, collate_fn
from model import create_sam_lora_model
from trainer import SAMTrainer
from utils import set_seed, get_device


# ============================================================================
# Example 1: Custom Data Augmentation
# ============================================================================

def create_custom_dataloader():
    """Create dataloader with custom augmentations"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    custom_transforms = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Medical image specific augmentations
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.5)
        ], p=0.3),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(p=0.8),
            A.HueSaturationValue(p=0.8),
            A.ColorJitter(p=0.8)
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
            A.MotionBlur(p=0.5)
        ], p=0.3),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    config = get_default_config()
    paths = config.data.get_full_paths()
    
    dataset = COCOSegmentationDataset(
        annotation_file=paths['train_ann'],
        image_dir=paths['train_img'],
        transforms=custom_transforms,
        image_size=1024
    )
    
    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )


# ============================================================================
# Example 2: Custom Loss with Boundary Emphasis
# ============================================================================

class BoundaryAwareLoss(nn.Module):
    """
    Loss function that emphasizes boundaries in segmentation.
    """
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def get_boundary_mask(self, mask, kernel_size=5):
        """Extract boundary from mask"""
        import torch.nn.functional as F
        
        # Erosion and dilation to find boundaries
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
        
        dilated = F.conv2d(mask.unsqueeze(1).float(), kernel, padding=kernel_size//2)
        eroded = -F.conv2d(1 - mask.unsqueeze(1).float(), kernel, padding=kernel_size//2)
        
        boundary = (dilated > 0).float() - (eroded > 0).float()
        return boundary.squeeze(1)
    
    def forward(self, pred, target):
        # Base BCE loss
        loss = self.bce(pred, target)
        
        # Get boundary regions
        boundary_mask = self.get_boundary_mask(target)
        
        # Weight loss more heavily on boundaries
        weighted_loss = loss * (1 + self.alpha * boundary_mask)
        
        return weighted_loss.mean()


# ============================================================================
# Example 3: Multi-Scale Training
# ============================================================================

def train_multiscale():
    """Train with multiple image scales"""
    config = get_default_config()
    device = get_device(config.device)
    set_seed(config.seed)
    
    # Create model
    model = create_sam_lora_model(config, device=device)
    
    # Different scales to train on
    scales = [512, 768, 1024]
    
    for epoch in range(config.training.num_epochs):
        # Randomly select scale for this epoch
        scale = np.random.choice(scales)
        print(f"Epoch {epoch + 1}: Training at scale {scale}x{scale}")
        
        # Update config for this scale
        config.data.image_size = scale
        
        # Create dataloaders for this scale
        # ... (implementation continues)


# ============================================================================
# Example 4: Progressive Training (Freezing/Unfreezing)
# ============================================================================

def progressive_training():
    """
    Progressive training strategy:
    1. Train only LoRA layers (frozen encoder)
    2. Unfreeze and fine-tune entire model
    """
    config = get_default_config()
    device = get_device(config.device)
    set_seed(config.seed)
    
    from dataset import create_dataloaders
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(config)
    
    # Phase 1: Train with frozen encoder (first 50 epochs)
    print("\n" + "="*60)
    print("Phase 1: Training with frozen encoder")
    print("="*60)
    
    config.model.freeze_image_encoder = True
    model_phase1 = create_sam_lora_model(config, device=device)
    
    config.training.num_epochs = 50
    trainer_phase1 = SAMTrainer(model_phase1, train_loader, val_loader, config, device)
    trainer_phase1.train()
    
    # Phase 2: Fine-tune entire model (next 50 epochs)
    print("\n" + "="*60)
    print("Phase 2: Fine-tuning entire model")
    print("="*60)
    
    config.model.freeze_image_encoder = False
    config.training.learning_rate = 1e-5  # Lower learning rate for fine-tuning
    model_phase2 = create_sam_lora_model(config, device=device)
    
    # Load weights from phase 1
    model_phase2.sam.load_state_dict(model_phase1.sam.state_dict())
    
    config.training.num_epochs = 50
    trainer_phase2 = SAMTrainer(model_phase2, train_loader, val_loader, config, device)
    trainer_phase2.train()


# ============================================================================
# Example 5: Model Ensemble for Inference
# ============================================================================

class SAMEnsemble:
    """
    Ensemble multiple SAM models for more robust predictions.
    """
    
    def __init__(self, model_paths, device='cuda'):
        self.models = []
        self.device = device
        
        config = get_default_config()
        
        for path in model_paths:
            model = create_sam_lora_model(config, device=device)
            model.load_checkpoint(path)
            model.eval()
            self.models.append(model)
    
    def predict(self, image, **kwargs):
        """
        Predict using ensemble averaging.
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred, iou = model(image, **kwargs)
                predictions.append(torch.sigmoid(pred))
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        # Compute uncertainty as standard deviation
        uncertainty = torch.stack(predictions).std(dim=0)
        
        return ensemble_pred, uncertainty


def use_ensemble():
    """Example of using ensemble for inference"""
    model_paths = [
        'outputs/exp1/checkpoints/best_model.pt',
        'outputs/exp2/checkpoints/best_model.pt',
        'outputs/exp3/checkpoints/best_model.pt'
    ]
    
    ensemble = SAMEnsemble(model_paths)
    
    # Load test image
    # image = ...
    
    # Get prediction and uncertainty
    # pred, uncertainty = ensemble.predict(image, ...)
    
    print("Ensemble ready for inference")


# ============================================================================
# Example 6: Test-Time Augmentation (TTA)
# ============================================================================

class TestTimeAugmentation:
    """
    Apply test-time augmentation for more robust predictions.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def predict_with_tta(self, image, **kwargs):
        """
        Predict with multiple augmentations and average results.
        """
        predictions = []
        
        # Original
        with torch.no_grad():
            pred, _ = self.model(image, **kwargs)
            predictions.append(torch.sigmoid(pred))
        
        # Horizontal flip
        image_flipped_h = torch.flip(image, dims=[3])
        with torch.no_grad():
            pred, _ = self.model(image_flipped_h, **kwargs)
            pred = torch.flip(pred, dims=[3])
            predictions.append(torch.sigmoid(pred))
        
        # Vertical flip
        image_flipped_v = torch.flip(image, dims=[2])
        with torch.no_grad():
            pred, _ = self.model(image_flipped_v, **kwargs)
            pred = torch.flip(pred, dims=[2])
            predictions.append(torch.sigmoid(pred))
        
        # Both flips
        image_flipped_both = torch.flip(image, dims=[2, 3])
        with torch.no_grad():
            pred, _ = self.model(image_flipped_both, **kwargs)
            pred = torch.flip(pred, dims=[2, 3])
            predictions.append(torch.sigmoid(pred))
        
        # Average all predictions
        tta_pred = torch.stack(predictions).mean(dim=0)
        
        return tta_pred


# ============================================================================
# Example 7: Attention Visualization
# ============================================================================

def visualize_attention_maps(model, image):
    """
    Visualize attention maps from the image encoder.
    """
    import matplotlib.pyplot as plt
    
    # Get image embeddings and attention maps
    model.eval()
    with torch.no_grad():
        # This requires modifying the model to return attention weights
        # embeddings, attention_weights = model.sam.image_encoder(image, return_attention=True)
        pass
    
    # Visualize attention maps
    # fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    # ...


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print("SAM LoRA Advanced Usage Examples")
    print("="*60)
    print("\nThis script demonstrates advanced usage patterns.")
    print("Uncomment the function you want to run:\n")
    
    # Example 1: Custom augmentations
    # train_loader = create_custom_dataloader()
    # print(f"Created custom dataloader with {len(train_loader)} batches")
    
    # Example 2: Custom loss
    # loss_fn = BoundaryAwareLoss(alpha=0.5)
    # print("Created custom boundary-aware loss")
    
    # Example 3: Multi-scale training
    # train_multiscale()
    
    # Example 4: Progressive training
    # progressive_training()
    
    # Example 5: Model ensemble
    # use_ensemble()
    
    # Example 6: Test-time augmentation
    # tta = TestTimeAugmentation(model)
    # print("TTA ready")
    
    print("\nExamples are ready to use!")
    print("Edit this file to uncomment and run specific examples.")
