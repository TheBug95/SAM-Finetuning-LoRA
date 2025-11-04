"""
Script to export trained SAM LoRA model to HuggingFace Hub.
"""
import argparse
import os
import sys
from pathlib import Path
import torch
from huggingface_hub import HfApi, create_repo

# Add Core Modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Core Modules"))

from model import create_sam_lora_model
from config import get_default_config


def export_to_huggingface(
    checkpoint_path: str,
    repo_name: str,
    model_type: str = "vit_b",
    private: bool = False,
    commit_message: str = "Upload SAM LoRA model"
):
    """
    Export trained model to HuggingFace Hub.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        repo_name: Name of the HuggingFace repository (e.g., "username/sam-lora-cataract")
        model_type: SAM model type
        private: Whether the repository should be private
        commit_message: Commit message for the upload
    """
    print("\n" + "="*60)
    print("Exporting Model to HuggingFace Hub")
    print("="*60)
    
    # Create config
    config = get_default_config()
    config.model.model_type = model_type
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = create_sam_lora_model(config, device="cpu")
    model.load_checkpoint(checkpoint_path, load_optimizer=False)
    
    # Create temporary directory for export
    export_dir = Path("./huggingface_export")
    export_dir.mkdir(exist_ok=True)
    
    # Save model state dict
    model_path = export_dir / "pytorch_model.bin"
    torch.save(model.sam.state_dict(), model_path)
    print(f"✓ Saved model to {model_path}")
    
    # Create model card
    model_card = f"""---
tags:
- image-segmentation
- medical
- cataract
- sam
- lora
- segment-anything
license: apache-2.0
---

# SAM LoRA - Cataract Segmentation

This model is a fine-tuned version of Segment Anything Model (SAM) using LoRA (Low-Rank Adaptation) for cataract segmentation.

## Model Details

- **Base Model**: SAM {model_type.upper()}
- **Adaptation Method**: LoRA
- **Task**: Medical Image Segmentation (Cataract)
- **Training Data**: COCO format cataract dataset

## Usage

```python
import torch
from segment_anything import sam_model_registry
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(repo_id="{repo_name}", filename="pytorch_model.bin")

# Load base SAM
sam = sam_model_registry["{model_type}"]()

# Load fine-tuned weights
sam.load_state_dict(torch.load(model_path))
sam.eval()

# Use for inference
# ... your inference code
```

## Training

This model was trained using:
- Mixed precision training (AMP)
- Combined loss: Focal + Dice + IoU
- LoRA for efficient fine-tuning
- Early stopping
- Data augmentation

## Citation

If you use this model, please cite the original SAM paper:

```bibtex
@article{{kirillov2023segment,
  title={{Segment Anything}},
  author={{Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{{'a}}r, Piotr and Girshick, Ross}},
  journal={{arXiv:2304.02643}},
  year={{2023}}
}}
```

## License

Apache 2.0
"""
    
    model_card_path = export_dir / "README.md"
    with open(model_card_path, 'w') as f:
        f.write(model_card)
    print(f"✓ Created model card")
    
    # Create config.json
    config_dict = {
        "model_type": model_type,
        "architecture": "SAM with LoRA",
        "task": "image-segmentation",
        "domain": "medical",
    }
    
    import json
    config_path = export_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Created config.json")
    
    # Upload to HuggingFace
    print(f"\nUploading to HuggingFace Hub: {repo_name}")
    
    try:
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            create_repo(repo_name, private=private, exist_ok=True)
            print(f"✓ Created/verified repository")
        except Exception as e:
            print(f"Note: {e}")
        
        # Upload files
        api.upload_folder(
            folder_path=str(export_dir),
            repo_id=repo_name,
            commit_message=commit_message
        )
        
        print(f"\n✅ Successfully exported model to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"\n❌ Error uploading to HuggingFace: {e}")
        print("\nMake sure you are logged in:")
        print("  huggingface-cli login")
        return False
    
    # Cleanup
    import shutil
    shutil.rmtree(export_dir)
    print(f"✓ Cleaned up temporary files")
    
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='Export SAM LoRA model to HuggingFace')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--repo_name', type=str, required=True,
                       help='HuggingFace repository name (e.g., username/sam-lora-cataract)')
    parser.add_argument('--model_type', type=str, default='vit_b',
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model type')
    parser.add_argument('--private', action='store_true',
                       help='Make repository private')
    parser.add_argument('--commit_message', type=str,
                       default='Upload SAM LoRA model',
                       help='Commit message')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    success = export_to_huggingface(
        checkpoint_path=args.checkpoint,
        repo_name=args.repo_name,
        model_type=args.model_type,
        private=args.private,
        commit_message=args.commit_message
    )
    
    if success:
        print("\n" + "="*60)
        print("Export Complete!")
        print("="*60)
        print(f"\nYour model is now available at:")
        print(f"https://huggingface.co/{args.repo_name}")
    else:
        print("\n❌ Export failed!")


if __name__ == '__main__':
    main()
