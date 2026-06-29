"""
Script de prueba para verificar la carga del dataset de glaucoma (Todo Dataset).

Este script verifica que el GlaucomaDataset funcione correctamente sin necesidad
de ejecutar el entrenamiento completo.

Configuracion:
    - Entrena SOLO con la mascara del optic disc
    - Incluye TODAS las imagenes de train/val/test (normales + glaucoma)
    - Las imagenes normales usan mascara vacia
"""
import sys
from pathlib import Path

# Add Core Modules to path
sys.path.insert(0, str(Path(__file__).parent / "Core Modules"))

from dataset import GlaucomaDataset, create_glaucoma_dataloaders, collate_fn
from config import get_default_config
from torch.utils.data import DataLoader


def test_default_config():
    """Verifica que la configuracion por defecto use solo disco e incluya todas las imagenes"""
    print("=" * 80)
    print("VERIFICACION DE CONFIGURACION POR DEFECTO")
    print("=" * 80)
    
    config = get_default_config()
    print(f"Dataset type: {config.data.dataset_type}")
    print(f"Glaucoma root: {config.glaucoma_data.dataset_root}")
    print(f"Mask format: {config.glaucoma_data.mask_format}")
    print(f"Mask suffixes (por defecto): {config.glaucoma_data.mask_suffixes}")
    print(f"Classes: {config.glaucoma_data.class_names}")
    print(f"use_only_labeled: {config.glaucoma_data.use_only_labeled}")
    
    assert config.glaucoma_data.mask_suffixes == ["disc"], \
        "ERROR: El default de mask_suffixes deberia ser ['disc']"
    assert config.glaucoma_data.class_names == ["optic_disc"], \
        "ERROR: El default de class_names deberia ser ['optic_disc']"
    assert config.glaucoma_data.use_only_labeled == False, \
        "ERROR: El default de use_only_labeled deberia ser False"
    
    print("\nConfiguracion por defecto correcta: solo optic disc + todas las imagenes")


def test_dataset_basic():
    """Prueba basica de carga del dataset con mascara de disco"""
    print("\n" + "=" * 80)
    print("PRUEBA BASICA DE GLAUCOMA DATASET (SOLO DISCO, TODAS LAS IMAGENES)")
    print("=" * 80)
    
    dataset = GlaucomaDataset(
        split_file="Todo Dataset/split.json",
        dataset_root="Todo Dataset",
        split="train",
        transforms=None,  # Usa resize manual
        image_size=1024,
        use_point_prompts=True,
        use_box_prompts=True,
        mask_format="npy",
        mask_suffixes=["disc"],  # Solo disco
        use_only_labeled=False   # Incluir imagenes normales con mascara vacia
    )
    
    print(f"\nDataset cargado: {len(dataset)} imagenes")
    
    # Contar cuantas tienen mascara y cuantas no
    with_mask = sum(1 for item in dataset.items if item['has_masks'])
    without_mask = len(dataset) - with_mask
    print(f"  Con mascara (glaucoma): {with_mask}")
    print(f"  Sin mascara (normal): {without_mask}")
    
    # Probar una muestra con mascara y una sin mascara
    samples_to_show = []
    for i in range(len(dataset)):
        if len(samples_to_show) < 2:
            samples_to_show.append(i)
            continue
        # Buscar una sin mascara si no la tenemos
        has_mask = dataset.items[i]['has_masks']
        if not has_mask and all(dataset.items[j]['has_masks'] for j in samples_to_show):
            samples_to_show[1] = i
            break
    
    for i in samples_to_show:
        sample = dataset[i]
        has_mask = dataset.items[i]['has_masks']
        print(f"\nMuestra {i} ({'con mascara' if has_mask else 'SIN mascara - normal'}):")
        print(f"  image shape: {sample['image'].shape}")
        print(f"  masks shape: {sample['masks'].shape}")
        print(f"  mask area: {sample['masks'][0].sum().item():.0f}")
        print(f"  category_ids: {sample['category_ids'].tolist()}")
        print(f"  boxes shape: {sample['boxes'].shape}")
        print(f"  boxes: {sample['boxes'][0].tolist()}")
        print(f"  point_coords shape: {sample['point_coords'].shape}")
        print(f"  point_labels shape: {sample['point_labels'].shape}")
        print(f"  point_labels: {sample['point_labels'][0].tolist()}")
        print(f"  image_id: {sample['image_id'].item()}")
        # Verificar consistencia de prompts
        if has_mask:
            # Para imagen con mascara, debe haber al menos un punto positivo
            assert sample['point_labels'][0].sum().item() > 0, "ERROR: Imagen con mascara deberia tener puntos positivos"
        else:
            # Para imagen sin mascara, el punto debe ser negativo
            assert sample['point_labels'][0].sum().item() == 0, "ERROR: Imagen sin mascara deberia tener solo puntos negativos"
    
    print("\nPrueba basica exitosa!")


def test_dataloader():
    """Prueba del DataLoader con collate_fn"""
    print("\n" + "=" * 80)
    print("PRUEBA DE DATALOADER")
    print("=" * 80)
    
    config = get_default_config()
    config.glaucoma_data.dataset_root = "Todo Dataset"
    config.glaucoma_data.image_size = 1024
    config.glaucoma_data.mask_suffixes = ["disc"]  # Asegurar solo disco
    config.glaucoma_data.use_only_labeled = False  # Incluir normales
    config.training.batch_size = 2
    
    train_loader, val_loader, test_loader = create_glaucoma_dataloaders(
        config,
        num_workers=0  # Usar 0 para evitar problemas en Windows
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Probar un batch
    batch = next(iter(train_loader))
    print(f"\nBatch train:")
    print(f"  images shape: {batch['images'].shape}")
    print(f"  num muestras masks: {len(batch['masks'])}")
    print(f"  num muestras boxes: {len(batch['boxes'])}")
    print(f"  num muestras point_coords: {len(batch['point_coords'])}")
    
    # Verificar que cada muestra tenga solo 1 mascara
    for i, masks in enumerate(batch['masks']):
        assert masks.shape[0] == 1, f"ERROR: Muestra {i} deberia tener 1 mascara"
    
    print("\nPrueba de DataLoader exitosa!")


if __name__ == "__main__":
    test_default_config()
    test_dataset_basic()
    test_dataloader()
    print("\n" + "=" * 80)
    print("TODAS LAS PRUEBAS PASARON")
    print("=" * 80)
