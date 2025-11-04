#!/usr/bin/env python
"""
Quick start script for SAM LoRA training.
This script provides a simple interface to start training with sensible defaults.
"""
import subprocess
import sys
import os
from pathlib import Path


def check_checkpoint_exists():
    """Check if SAM checkpoint exists"""
    checkpoint_paths = [
        "checkpoints/sam_vit_b_01ec64.pth",
        "../checkpoints/sam_vit_b_01ec64.pth",
        "./sam_vit_b_01ec64.pth"
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    print("="*60)
    print("SAM LoRA Quick Start")
    print("="*60)
    
    # Check for checkpoint
    checkpoint = check_checkpoint_exists()
    
    if checkpoint is None:
        print("\n⚠️  SAM checkpoint not found!")
        print("\nPlease download SAM ViT-Base checkpoint from:")
        print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print("\nOr run:")
        print("mkdir -p checkpoints")
        print("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b_01ec64.pth")
        sys.exit(1)
    
    print(f"\n✓ Found checkpoint: {checkpoint}")
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Train with default settings")
    print("2. Run hyperparameter optimization with Optuna")
    print("3. Run inference on trained model")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        # Training
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        # Get path to train.py in Main Scripts folder
        script_dir = Path(__file__).parent.parent / "Main Scripts"
        train_script = script_dir / "train.py"
        
        cmd = [
            sys.executable, str(train_script),
            "--checkpoint", checkpoint,
            "--batch_size", "4",
            "--num_epochs", "100",
            "--lr", "1e-4",
            "--lora_rank", "8",
            "--lora_alpha", "16"
        ]
        
        print(f"\nCommand: {' '.join(cmd)}\n")
        subprocess.run(cmd)
    
    elif choice == "2":
        # Optuna
        print("\n" + "="*60)
        print("Starting Hyperparameter Optimization")
        print("="*60)
        
        n_trials = input("\nNumber of trials (default: 50): ").strip() or "50"
        
        # Get path to optuna_tuning.py in Main Scripts folder
        script_dir = Path(__file__).parent.parent / "Main Scripts"
        optuna_script = script_dir / "optuna_tuning.py"
        
        cmd = [
            sys.executable, str(optuna_script),
            "--checkpoint", checkpoint,
            "--n_trials", n_trials
        ]
        
        print(f"\nCommand: {' '.join(cmd)}\n")
        subprocess.run(cmd)
    
    elif choice == "3":
        # Inference
        print("\n" + "="*60)
        print("Running Inference")
        print("="*60)
        
        # Find trained model
        model_paths = []
        if os.path.exists("outputs"):
            for root, dirs, files in os.walk("outputs"):
                for file in files:
                    if file == "best_model.pt":
                        model_paths.append(os.path.join(root, file))
        
        if len(model_paths) == 0:
            print("\n⚠️  No trained models found!")
            print("Please train a model first.")
            sys.exit(1)
        
        print("\nAvailable models:")
        for i, path in enumerate(model_paths):
            print(f"{i+1}. {path}")
        
        if len(model_paths) == 1:
            model_choice = 0
        else:
            model_choice = int(input(f"\nSelect model (1-{len(model_paths)}): ").strip()) - 1
        
        model_path = model_paths[model_choice]
        
        split = input("\nDataset split (train/valid/test, default: test): ").strip() or "test"
        
        # Get path to inference.py in Main Scripts folder
        script_dir = Path(__file__).parent.parent / "Main Scripts"
        inference_script = script_dir / "inference.py"
        
        cmd = [
            sys.executable, str(inference_script),
            "--checkpoint", model_path,
            "--split", split,
            "--save_visualizations"
        ]
        
        print(f"\nCommand: {' '.join(cmd)}\n")
        subprocess.run(cmd)
    
    elif choice == "4":
        print("\nGoodbye!")
        sys.exit(0)
    
    else:
        print("\n⚠️  Invalid choice!")
        sys.exit(1)


if __name__ == "__main__":
    main()
