"""
Main training script for SAM LoRA fine-tuning.
"""
import argparse
import os
from pathlib import Path

import sys
from pathlib import Path
# Add Core Modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Core Modules"))

import torch

from config import Config, get_default_config
from dataset import create_dataloaders
from model import create_sam_lora_model
from trainer import SAMTrainer
from utils import set_seed, get_device, save_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train SAM with LoRA')
    
    # Data
    parser.add_argument('--data_root', type=str, 
                       default='../Cataract COCO Segmentation/Cataract COCO Segmentation',
                       help='Root directory of COCO dataset')
    parser.add_argument('--image_size', type=int, default=1024,
                       help='Input image size')
    
    # Model
    parser.add_argument('--model_type', type=str, default='vit_b',
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model type')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to pretrained SAM checkpoint')
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout rate')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    # Loss weights
    parser.add_argument('--focal_weight', type=float, default=20.0,
                       help='Focal loss weight')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                       help='Dice loss weight')
    parser.add_argument('--iou_weight', type=float, default=1.0,
                       help='IoU loss weight')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='sam_lora_cataract',
                       help='Experiment name')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='sam-lora-finetuning',
                       help='W&B project name')
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create config
    config = get_default_config()
    
    # Update config with command line arguments
    config.data.coco_root = args.data_root
    config.data.image_size = args.image_size
    config.data.num_workers = args.num_workers
    
    config.model.model_type = args.model_type
    config.model.checkpoint_path = args.checkpoint
    config.model.lora_rank = args.lora_rank
    config.model.lora_alpha = args.lora_alpha
    config.model.lora_dropout = args.lora_dropout
    
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.num_epochs
    config.training.learning_rate = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.focal_loss_weight = args.focal_weight
    config.training.dice_loss_weight = args.dice_weight
    config.training.iou_loss_weight = args.iou_weight
    
    config.seed = args.seed
    config.device = args.device
    config.output_dir = args.output_dir
    config.experiment_name = args.experiment_name
    config.resume_from = args.resume_from
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    
    # Re-initialize to create directories
    config.__post_init__()
    
    # Set seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    
    # Print configuration
    print("\n" + "="*50)
    print("Configuration:")
    print("="*50)
    print(f"Data root: {config.data.coco_root}")
    print(f"Model type: {config.model.model_type}")
    print(f"LoRA rank: {config.model.lora_rank}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Number of epochs: {config.training.num_epochs}")
    print(f"Device: {device}")
    print(f"Output directory: {config.experiment_dir}")
    print("="*50 + "\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        num_workers=config.data.num_workers
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Create model
    print("Creating model...")
    model = create_sam_lora_model(config, device=device)
    print()
    
    # Resume from checkpoint if specified
    if config.resume_from is not None:
        print(f"Resuming from checkpoint: {config.resume_from}")
        start_epoch, optimizer_state = model.load_checkpoint(
            config.resume_from,
            load_optimizer=True
        )
        print()
    
    # Create trainer
    print("Creating trainer...")
    trainer = SAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=config.use_wandb
    )
    
    # Load optimizer state if resuming
    if config.resume_from is not None and optimizer_state is not None:
        trainer.optimizer.load_state_dict(optimizer_state)
        trainer.current_epoch = start_epoch
    
    # Train
    history = trainer.train()
    
    # Save training history
    history_path = os.path.join(config.experiment_dir, 'training_history.json')
    save_metrics(history, history_path)
    print(f"Training history saved to {history_path}")
    
    # Save final model
    final_path = os.path.join(config.experiment_dir, 'final_model.pt')
    model.save_checkpoint(final_path, epoch=config.training.num_epochs)
    print(f"Final model saved to {final_path}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
