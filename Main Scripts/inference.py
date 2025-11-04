"""
Inference script for testing trained SAM LoRA model.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
import json

# Add Core Modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Core Modules"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from PIL import Image

from config import Config, get_default_config
from dataset import COCOSegmentationDataset, get_transforms, collate_fn
from model import create_sam_lora_model
from utils import (
    set_seed, get_device, calculate_iou, calculate_dice,
    visualize_predictions, save_metrics, AverageMeter
)


class SAMInference:
    """
    Inference class for trained SAM LoRA model.
    """
    
    def __init__(
        self,
        model,
        device: str = "cuda",
        threshold: float = 0.5
    ):
        """
        Args:
            model: Trained SAM LoRA model
            device: Device to run inference on
            threshold: Threshold for binary mask prediction
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.eval()
    
    def predict_single_image(
        self,
        image: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        multimask_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Predict mask for a single image.
        
        Args:
            image: [1, 3, H, W] or [3, H, W] input image
            point_coords: Point prompts
            point_labels: Point labels
            boxes: Box prompts
            multimask_output: Whether to output multiple masks
        
        Returns:
            Dictionary with 'masks' and 'iou_predictions'
        """
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            image = image.to(self.device)
            
            if point_coords is not None:
                point_coords = point_coords.to(self.device)
            if point_labels is not None:
                point_labels = point_labels.to(self.device)
            if boxes is not None:
                boxes = boxes.to(self.device)
            
            masks, iou_predictions = self.model(
                image,
                point_coords=point_coords,
                point_labels=point_labels,
                boxes=boxes,
                multimask_output=multimask_output
            )
            
            # Apply threshold
            masks = (torch.sigmoid(masks) > self.threshold).float()
        
        return {
            'masks': masks.cpu(),
            'iou_predictions': iou_predictions.cpu()
        }
    
    def predict_batch(
        self,
        dataloader: DataLoader,
        save_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Run inference on a batch of images.
        
        Args:
            dataloader: DataLoader with test images
            save_dir: Directory to save visualizations (optional)
        
        Returns:
            Dictionary with evaluation metrics
        """
        iou_meter = AverageMeter()
        dice_meter = AverageMeter()
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            images = batch['images'].to(self.device)
            masks_list = batch['masks']
            boxes_list = batch['boxes']
            point_coords_list = batch['point_coords']
            point_labels_list = batch['point_labels']
            image_ids = batch['image_ids']
            
            # Process each image
            for i in range(len(images)):
                image = images[i:i+1]
                true_masks = masks_list[i]
                
                boxes = boxes_list[i] if boxes_list[i] is not None else None
                point_coords = point_coords_list[i] if point_coords_list[i] is not None else None
                point_labels = point_labels_list[i] if point_labels_list[i] is not None else None
                
                image_results = {
                    'image_id': image_ids[i].item(),
                    'objects': []
                }
                
                # Process each object/mask
                for obj_idx in range(len(true_masks)):
                    true_mask = true_masks[obj_idx:obj_idx+1]
                    
                    obj_boxes = boxes[obj_idx:obj_idx+1] if boxes is not None else None
                    obj_points = point_coords[obj_idx:obj_idx+1] if point_coords is not None else None
                    obj_labels = point_labels[obj_idx:obj_idx+1] if point_labels is not None else None
                    
                    # Predict
                    pred = self.predict_single_image(
                        image,
                        point_coords=obj_points,
                        point_labels=obj_labels,
                        boxes=obj_boxes,
                        multimask_output=False
                    )
                    
                    pred_mask = pred['masks'][0, 0]  # [H, W]
                    true_mask_2d = true_mask[0]  # [H, W]
                    
                    # Calculate metrics
                    iou = calculate_iou(pred_mask, true_mask_2d, threshold=self.threshold)
                    dice = calculate_dice(pred_mask, true_mask_2d, threshold=self.threshold)
                    
                    iou_meter.update(iou)
                    dice_meter.update(dice)
                    
                    image_results['objects'].append({
                        'object_id': obj_idx,
                        'iou': iou,
                        'dice': dice
                    })
                    
                    # Save visualization for first object of first few images
                    if save_dir is not None and batch_idx < 10 and obj_idx == 0:
                        # Convert image to numpy
                        img_np = image[0].cpu().numpy().transpose(1, 2, 0)
                        # Denormalize
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = (img_np * std + mean) * 255
                        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                        
                        pred_mask_np = pred_mask.numpy()
                        true_mask_np = true_mask_2d.numpy()
                        
                        save_path = save_dir / f"image_{image_ids[i].item()}_obj_{obj_idx}.png"
                        visualize_predictions(
                            img_np,
                            pred_mask_np,
                            true_mask_np,
                            save_path=str(save_path)
                        )
                
                results.append(image_results)
        
        # Aggregate metrics
        metrics = {
            'mean_iou': iou_meter.avg,
            'mean_dice': dice_meter.avg,
            'num_images': len(results),
            'num_objects': iou_meter.count
        }
        
        return metrics, results


def parse_args():
    parser = argparse.ArgumentParser(description='SAM LoRA Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str,
                       default='../Cataract COCO Segmentation/Cataract COCO Segmentation',
                       help='Root directory of COCO dataset')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--model_type', type=str, default='vit_b',
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model type')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask prediction')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Output directory for results')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print("SAM LoRA Inference")
    print("="*50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print("="*50 + "\n")
    
    # Create config
    config = get_default_config()
    config.data.coco_root = args.data_root
    config.model.model_type = args.model_type
    config.device = args.device
    
    # Load model
    print("Loading model...")
    model = create_sam_lora_model(config, device=device)
    model.load_checkpoint(args.checkpoint, load_optimizer=False)
    print()
    
    # Create dataset
    print(f"Loading {args.split} dataset...")
    paths = config.data.get_full_paths()
    
    if args.split == 'train':
        ann_file = paths['train_ann']
        img_dir = paths['train_img']
    elif args.split == 'valid':
        ann_file = paths['val_ann']
        img_dir = paths['val_img']
    else:  # test
        ann_file = paths['test_ann']
        img_dir = paths['test_img']
    
    dataset = COCOSegmentationDataset(
        annotation_file=ann_file,
        image_dir=img_dir,
        transforms=get_transforms(config.data.image_size, is_train=False),
        image_size=config.data.image_size,
        use_point_prompts=True,
        use_box_prompts=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Loaded {len(dataset)} images\n")
    
    # Create inference object
    inference = SAMInference(
        model=model,
        device=device,
        threshold=args.threshold
    )
    
    # Run inference
    print("Running inference...")
    viz_dir = output_dir / 'visualizations' if args.save_visualizations else None
    metrics, results = inference.predict_batch(dataloader, save_dir=viz_dir)
    
    # Print results
    print("\n" + "="*50)
    print("Inference Results")
    print("="*50)
    print(f"Number of images: {metrics['num_images']}")
    print(f"Number of objects: {metrics['num_objects']}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {metrics['mean_dice']:.4f}")
    print("="*50 + "\n")
    
    # Save results
    metrics_path = output_dir / 'metrics.json'
    save_metrics(metrics, str(metrics_path))
    print(f"Metrics saved to {metrics_path}")
    
    detailed_results_path = output_dir / 'detailed_results.json'
    with open(detailed_results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Detailed results saved to {detailed_results_path}")
    
    if args.save_visualizations:
        print(f"Visualizations saved to {viz_dir}")
    
    print("\nInference complete!")


if __name__ == '__main__':
    main()
