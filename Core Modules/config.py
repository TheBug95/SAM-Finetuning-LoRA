"""
Configuration file for SAM LoRA fine-tuning.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Data configuration
    
    Note: The dataset uses binary classification:
        - Class 0: No cataract (normal)
        - Class 1: Cataract (includes cataract, mild, severe from original annotations)
    """
    coco_root: str = "../Cataract COCO Segmentation/Cataract COCO Segmentation"
    train_ann_file: str = "train/_annotations.coco.json"
    val_ann_file: str = "valid/_annotations.coco.json"
    test_ann_file: str = "test/_annotations.coco.json"
    train_img_dir: str = "train"
    val_img_dir: str = "valid"
    test_img_dir: str = "test"
    image_size: int = 1024  # SAM's default input size
    num_workers: int = 4
    
    # Class information
    num_classes: int = 2  # Binary classification: cataract vs no cataract
    class_names: List[str] = field(default_factory=lambda: ["no_cataract", "cataract"])
    
    def get_full_paths(self):
        """Get full paths for annotations and images"""
        return {
            'train_ann': os.path.join(self.coco_root, self.train_ann_file),
            'val_ann': os.path.join(self.coco_root, self.val_ann_file),
            'test_ann': os.path.join(self.coco_root, self.test_ann_file),
            'train_img': os.path.join(self.coco_root, self.train_img_dir),
            'val_img': os.path.join(self.coco_root, self.val_img_dir),
            'test_img': os.path.join(self.coco_root, self.test_img_dir)
        }


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = "vit_b"  # SAM ViT-Base
    checkpoint_path: Optional[str] = None  # Path to pretrained SAM checkpoint
    freeze_image_encoder: bool = False  # Whether to freeze image encoder
    freeze_prompt_encoder: bool = True  # Whether to freeze prompt encoder
    freeze_mask_decoder: bool = False  # Whether to freeze mask decoder
    
    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "qkv",  # Attention query, key, value
        "proj"  # Projection layers
    ])


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True  # Use AMP
    
    # Optimizer
    optimizer: str = "adamw"  # adamw or adam
    betas: tuple = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = "cosine"  # cosine, linear, or step
    
    # Loss weights
    focal_loss_weight: float = 20.0
    dice_loss_weight: float = 1.0
    iou_loss_weight: float = 1.0
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Checkpointing
    save_every: int = 5  # Save checkpoint every N epochs
    keep_last_n: int = 3  # Keep last N checkpoints
    
    # Logging
    log_every: int = 10  # Log every N batches


@dataclass
class OptunaConfig:
    """Optuna hyperparameter tuning configuration"""
    n_trials: int = 50
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
    study_name: str = "sam_lora_optimization"
    storage: Optional[str] = None  # SQLite database for study storage
    
    # Search space
    lr_range: tuple = (1e-5, 1e-3)
    batch_size_choices: List[int] = field(default_factory=lambda: [2, 4, 8])
    lora_rank_choices: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    lora_alpha_choices: List[int] = field(default_factory=lambda: [8, 16, 32])
    weight_decay_range: tuple = (1e-5, 1e-1)


@dataclass
class Config:
    """Main configuration"""
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    
    # General
    seed: int = 42
    device: str = "cuda"  # cuda or cpu
    output_dir: str = "./outputs"
    experiment_name: str = "sam_lora_cataract"
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    
    # Wandb logging (optional)
    use_wandb: bool = False
    wandb_project: str = "sam-lora-finetuning"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        self.experiment_dir = experiment_dir


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
