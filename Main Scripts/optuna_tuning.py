"""
Optuna hyperparameter optimization for SAM LoRA.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict

# Add Core Modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Core Modules"))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

from config import Config, get_default_config
from dataset import create_dataloaders
from model import create_sam_lora_model
from trainer import SAMTrainer
from utils import set_seed, get_device


def objective(trial: optuna.Trial, base_config: Config, device: torch.device) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration
        device: Device to train on
    
    Returns:
        Validation loss (to minimize)
    """
    # Sample hyperparameters
    config = get_default_config()
    
    # Copy base config settings
    config.data = base_config.data
    config.seed = base_config.seed
    config.device = base_config.device
    config.output_dir = base_config.output_dir
    config.model.model_type = base_config.model.model_type
    config.model.checkpoint_path = base_config.model.checkpoint_path
    
    # Sample hyperparameters
    config.training.learning_rate = trial.suggest_float(
        'learning_rate',
        base_config.optuna.lr_range[0],
        base_config.optuna.lr_range[1],
        log=True
    )
    
    config.training.batch_size = trial.suggest_categorical(
        'batch_size',
        base_config.optuna.batch_size_choices
    )
    
    config.model.lora_rank = trial.suggest_categorical(
        'lora_rank',
        base_config.optuna.lora_rank_choices
    )
    
    config.model.lora_alpha = trial.suggest_categorical(
        'lora_alpha',
        base_config.optuna.lora_alpha_choices
    )
    
    config.training.weight_decay = trial.suggest_float(
        'weight_decay',
        base_config.optuna.weight_decay_range[0],
        base_config.optuna.weight_decay_range[1],
        log=True
    )
    
    config.training.focal_loss_weight = trial.suggest_float(
        'focal_weight',
        10.0, 30.0
    )
    
    config.training.dice_loss_weight = trial.suggest_float(
        'dice_weight',
        0.5, 2.0
    )
    
    config.training.iou_loss_weight = trial.suggest_float(
        'iou_weight',
        0.5, 2.0
    )
    
    # Set experiment name for this trial
    config.experiment_name = f"optuna_trial_{trial.number}"
    config.__post_init__()
    
    # Set seed
    set_seed(config.seed)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    print(f"Learning rate: {config.training.learning_rate:.6f}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"LoRA rank: {config.model.lora_rank}")
    print(f"LoRA alpha: {config.model.lora_alpha}")
    print(f"Weight decay: {config.training.weight_decay:.6f}")
    print(f"Focal weight: {config.training.focal_loss_weight:.2f}")
    print(f"Dice weight: {config.training.dice_loss_weight:.2f}")
    print(f"IoU weight: {config.training.iou_loss_weight:.2f}")
    print(f"{'='*60}\n")
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        config,
        num_workers=config.data.num_workers
    )
    
    # Create model
    model = create_sam_lora_model(config, device=device)
    
    # Create trainer
    trainer = SAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=False  # Disable wandb for Optuna trials
    )
    
    # Train for a limited number of epochs
    max_epochs = min(20, base_config.training.num_epochs)
    config.training.num_epochs = max_epochs
    
    best_val_loss = float('inf')
    
    for epoch in range(max_epochs):
        trainer.current_epoch = epoch
        
        # Train
        train_metrics = trainer.train_epoch()
        
        # Validate
        val_metrics = trainer.validate()
        
        # Update scheduler
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        
        val_loss = val_metrics['val_loss']
        
        # Track best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        # Report intermediate value for pruning
        trial.report(val_loss, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch}")
            raise optuna.TrialPruned()
        
        print(f"Epoch {epoch + 1}/{max_epochs} - "
              f"Train Loss: {train_metrics['train_loss']:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val IoU: {val_metrics['val_iou']:.4f}")
    
    # Clean up
    del model
    del trainer
    torch.cuda.empty_cache()
    
    return best_val_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization for SAM LoRA')
    
    parser.add_argument('--data_root', type=str,
                       default='../Cataract COCO Segmentation/Cataract COCO Segmentation',
                       help='Root directory of COCO dataset')
    parser.add_argument('--model_type', type=str, default='vit_b',
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model type')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to pretrained SAM checkpoint')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel jobs')
    parser.add_argument('--study_name', type=str, default='sam_lora_optimization',
                       help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None,
                       help='Optuna storage (e.g., sqlite:///optuna.db)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create base config
    config = get_default_config()
    config.data.coco_root = args.data_root
    config.model.model_type = args.model_type
    config.model.checkpoint_path = args.checkpoint
    config.seed = args.seed
    config.device = args.device
    config.output_dir = args.output_dir
    config.optuna.n_trials = args.n_trials
    config.optuna.study_name = args.study_name
    config.optuna.storage = args.storage
    
    # Get device
    device = get_device(config.device)
    
    # Create study
    print("\n" + "="*60)
    print("Starting Optuna Hyperparameter Optimization")
    print("="*60)
    print(f"Study name: {config.optuna.study_name}")
    print(f"Number of trials: {config.optuna.n_trials}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Create sampler and pruner
    sampler = TPESampler(
        seed=config.seed,
        n_startup_trials=config.optuna.n_startup_trials
    )
    
    pruner = MedianPruner(
        n_startup_trials=config.optuna.n_startup_trials,
        n_warmup_steps=config.optuna.n_warmup_steps
    )
    
    # Create or load study
    study = optuna.create_study(
        study_name=config.optuna.study_name,
        storage=config.optuna.storage,
        load_if_exists=True,
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, config, device),
        n_trials=config.optuna.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Save results
    results_dir = os.path.join(config.output_dir, 'optuna_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save best parameters
    import json
    best_params_path = os.path.join(results_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Best parameters saved to {best_params_path}")
    
    # Save study results
    df = study.trials_dataframe()
    df_path = os.path.join(results_dir, 'trials.csv')
    df.to_csv(df_path, index=False)
    print(f"Trial results saved to {df_path}")
    
    # Create optimization history plot
    try:
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.savefig(os.path.join(results_dir, 'optimization_history.png'), dpi=150, bbox_inches='tight')
        print(f"Optimization history plot saved")
        
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.savefig(os.path.join(results_dir, 'param_importances.png'), dpi=150, bbox_inches='tight')
        print(f"Parameter importances plot saved")
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
    
    print("\nOptuna optimization complete!")


if __name__ == '__main__':
    main()
