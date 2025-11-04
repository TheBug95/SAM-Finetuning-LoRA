"""
Dataset module for loading COCO format data for SAM training.
"""
import os
import json
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as coco_mask
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class COCOSegmentationDataset(Dataset):
    """
    COCO format dataset for segmentation with SAM.
    """
    
    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        transforms: Optional[A.Compose] = None,
        image_size: int = 1024,
        use_point_prompts: bool = True,
        use_box_prompts: bool = True,
        num_points: int = 5
    ):
        """
        Args:
            annotation_file: Path to COCO annotation JSON file
            image_dir: Directory containing images
            transforms: Albumentations transforms
            image_size: Target image size (SAM default is 1024)
            use_point_prompts: Whether to generate point prompts
            use_box_prompts: Whether to generate box prompts
            num_points: Number of point prompts to generate per mask
        """
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_size = image_size
        self.use_point_prompts = use_point_prompts
        self.use_box_prompts = use_box_prompts
        self.num_points = num_points
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image id to annotations mapping
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Filter images that have annotations
        self.image_ids = [img_id for img_id in self.images.keys() 
                         if img_id in self.img_to_anns]
        
        print(f"Loaded {len(self.image_ids)} images with annotations")
        print(f"Categories: {[cat['name'] for cat in self.categories.values()]}")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _decode_rle(self, rle: Dict, height: int, width: int) -> np.ndarray:
        """Decode COCO RLE to binary mask"""
        if isinstance(rle, list):
            # Polygon format
            mask = coco_mask.decode(coco_mask.frPyObjects(rle, height, width))
            if len(mask.shape) == 3:
                mask = mask.max(axis=2)
        else:
            # RLE format
            mask = coco_mask.decode(rle)
        return mask
    
    def _get_box_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """Get bounding box from binary mask"""
        pos = np.where(mask)
        if len(pos[0]) == 0:
            return np.array([0, 0, 0, 0], dtype=np.float32)
        
        ymin, ymax = pos[0].min(), pos[0].max()
        xmin, xmax = pos[1].min(), pos[1].max()
        return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
    
    def _get_points_from_mask(
        self, 
        mask: np.ndarray, 
        num_points: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points from mask as prompts.
        Returns positive and negative points.
        """
        h, w = mask.shape
        
        # Positive points (from mask)
        pos_coords = np.argwhere(mask > 0)
        if len(pos_coords) == 0:
            pos_points = np.array([[h//2, w//2]], dtype=np.float32)
        else:
            if len(pos_coords) < num_points:
                indices = np.random.choice(len(pos_coords), num_points, replace=True)
            else:
                indices = np.random.choice(len(pos_coords), num_points, replace=False)
            pos_points = pos_coords[indices]
        
        # Negative points (outside mask)
        neg_coords = np.argwhere(mask == 0)
        if len(neg_coords) > 0:
            if len(neg_coords) < num_points:
                indices = np.random.choice(len(neg_coords), num_points, replace=True)
            else:
                indices = np.random.choice(len(neg_coords), num_points, replace=False)
            neg_points = neg_coords[indices]
        else:
            neg_points = np.array([], dtype=np.float32).reshape(0, 2)
        
        # Convert to (x, y) format and ensure float32
        pos_points = pos_points[:, [1, 0]].astype(np.float32)  # Flip to x, y
        if len(neg_points) > 0:
            neg_points = neg_points[:, [1, 0]].astype(np.float32)
        
        return pos_points, neg_points
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - image: [3, H, W]
                - masks: [N, H, W] where N is number of objects
                - boxes: [N, 4] bounding boxes in xyxy format
                - point_coords: [N, P, 2] point prompts
                - point_labels: [N, P] labels for points (1=fg, 0=bg)
                - category_ids: [N] category IDs
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        annotations = self.img_to_anns[img_id]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Collect all masks for this image
        masks = []
        boxes = []
        point_coords_list = []
        point_labels_list = []
        category_ids = []
        
        for ann in annotations:
            # Decode mask
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], dict):
                    # RLE format
                    mask = self._decode_rle(ann['segmentation'], orig_h, orig_w)
                else:
                    # Polygon format
                    mask = self._decode_rle(ann['segmentation'], orig_h, orig_w)
                
                if mask.sum() == 0:  # Skip empty masks
                    continue
                
                masks.append(mask)
                
                # Get box prompt
                if self.use_box_prompts:
                    box = self._get_box_from_mask(mask)
                    boxes.append(box)
                
                # Get point prompts
                if self.use_point_prompts:
                    pos_points, neg_points = self._get_points_from_mask(
                        mask, self.num_points
                    )
                    # Combine positive and negative points
                    all_points = np.vstack([pos_points, neg_points]) if len(neg_points) > 0 else pos_points
                    labels = np.concatenate([
                        np.ones(len(pos_points)),
                        np.zeros(len(neg_points)) if len(neg_points) > 0 else np.array([])
                    ])
                    point_coords_list.append(all_points)
                    point_labels_list.append(labels)
                
                category_ids.append(ann['category_id'])
        
        if len(masks) == 0:
            # Handle images with no valid masks by returning dummy data
            masks = [np.zeros((orig_h, orig_w), dtype=np.uint8)]
            boxes = [np.array([0, 0, orig_w, orig_h], dtype=np.float32)]
            point_coords_list = [np.array([[orig_w//2, orig_h//2]], dtype=np.float32)]
            point_labels_list = [np.array([1], dtype=np.float32)]
            category_ids = [0]
        
        # Stack masks
        masks = np.stack(masks, axis=0)  # [N, H, W]
        
        # Apply transforms
        if self.transforms is not None:
            # Albumentations requires mask to be HxWxN for multiple masks
            transformed = self.transforms(
                image=image,
                masks=[masks[i] for i in range(len(masks))]
            )
            image = transformed['image']
            masks = np.stack(transformed['masks'], axis=0) if len(transformed['masks']) > 0 else masks
        else:
            # Default: resize to target size
            image = cv2.resize(image, (self.image_size, self.image_size))
            resized_masks = []
            for m in masks:
                resized_m = cv2.resize(
                    m.astype(np.uint8), 
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST
                )
                resized_masks.append(resized_m)
            masks = np.stack(resized_masks, axis=0)
            
            # Normalize image
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # Scale boxes and points to new size
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)  # Convert to float32 before scaling
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        if len(point_coords_list) > 0:
            for i in range(len(point_coords_list)):
                point_coords_list[i] = point_coords_list[i].astype(np.float32)  # Convert to float32
                point_coords_list[i][:, 0] *= scale_x
                point_coords_list[i][:, 1] *= scale_y
        
        # Convert to tensors
        result = {
            'image': image,
            'masks': torch.from_numpy(masks).float(),
            'category_ids': torch.tensor(category_ids, dtype=torch.long),
            'image_id': torch.tensor(img_id, dtype=torch.long)
        }
        
        if self.use_box_prompts and len(boxes) > 0:
            result['boxes'] = torch.from_numpy(np.array(boxes)).float()
        
        if self.use_point_prompts and len(point_coords_list) > 0:
            # Pad point sequences to same length
            max_points = max(len(pts) for pts in point_coords_list)
            padded_coords = []
            padded_labels = []
            for coords, labels in zip(point_coords_list, point_labels_list):
                pad_len = max_points - len(coords)
                if pad_len > 0:
                    coords = np.vstack([coords, np.zeros((pad_len, 2))])
                    labels = np.concatenate([labels, np.zeros(pad_len)])
                padded_coords.append(coords)
                padded_labels.append(labels)
            
            result['point_coords'] = torch.from_numpy(np.array(padded_coords)).float()
            result['point_labels'] = torch.from_numpy(np.array(padded_labels)).long()
        
        return result


def get_transforms(image_size: int = 1024, is_train: bool = True) -> A.Compose:
    """
    Get augmentation transforms.
    
    Args:
        image_size: Target image size
        is_train: Whether this is for training (applies augmentations)
    
    Returns:
        Albumentations compose object
    """
    if is_train:
        transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transforms


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable number of objects per image.
    """
    # Since each image can have different number of masks, we return a list
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'masks': [item['masks'] for item in batch],
        'boxes': [item.get('boxes', None) for item in batch],
        'point_coords': [item.get('point_coords', None) for item in batch],
        'point_labels': [item.get('point_labels', None) for item in batch],
        'category_ids': [item['category_ids'] for item in batch],
        'image_ids': torch.stack([item['image_id'] for item in batch])
    }


def create_dataloaders(
    config,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration object
        num_workers: Number of dataloader workers
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    paths = config.data.get_full_paths()
    
    # Training dataset
    train_dataset = COCOSegmentationDataset(
        annotation_file=paths['train_ann'],
        image_dir=paths['train_img'],
        transforms=get_transforms(config.data.image_size, is_train=True),
        image_size=config.data.image_size,
        use_point_prompts=True,
        use_box_prompts=True
    )
    
    # Validation dataset
    val_dataset = COCOSegmentationDataset(
        annotation_file=paths['val_ann'],
        image_dir=paths['val_img'],
        transforms=get_transforms(config.data.image_size, is_train=False),
        image_size=config.data.image_size,
        use_point_prompts=True,
        use_box_prompts=True
    )
    
    # Test dataset
    test_dataset = COCOSegmentationDataset(
        annotation_file=paths['test_ann'],
        image_dir=paths['test_img'],
        transforms=get_transforms(config.data.image_size, is_train=False),
        image_size=config.data.image_size,
        use_point_prompts=True,
        use_box_prompts=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for testing
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
