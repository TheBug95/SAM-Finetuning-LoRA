"""
Script de prueba para verificar el mapeo de categorías binarias.
Ejecuta este script para confirmar que el mapeo de 4 clases a 2 clases funciona correctamente.
"""
import sys
from pathlib import Path

# Add Core Modules to path
sys.path.insert(0, str(Path(__file__).parent / "Core Modules"))

from config import get_default_config
from dataset import COCOSegmentationDataset

def test_binary_mapping():
    """Prueba el mapeo de categorías binarias"""
    print("=" * 80)
    print("PRUEBA DE MAPEO DE CATEGORÍAS BINARIAS")
    print("=" * 80)
    
    # Get config
    config = get_default_config()
    paths = config.data.get_full_paths()
    
    print(f"\nConfiguración:")
    print(f"  - Número de clases: {config.data.num_classes}")
    print(f"  - Nombres de clases: {config.data.class_names}")
    
    # Try to load training dataset
    try:
        print(f"\nIntentando cargar dataset de entrenamiento...")
        print(f"  Archivo de anotaciones: {paths['train_ann']}")
        print(f"  Directorio de imágenes: {paths['train_img']}")
        print()
        
        train_dataset = COCOSegmentationDataset(
            annotation_file=paths['train_ann'],
            image_dir=paths['train_img'],
            image_size=1024,
            use_point_prompts=True,
            use_box_prompts=True
        )
        
        print(f"\n✓ Dataset cargado exitosamente!")
        print(f"  - Total de imágenes: {len(train_dataset)}")
        
        # Verificar mapeo
        print(f"\nVerificando mapeo de categorías:")
        print(f"  Categorías originales: {list(train_dataset.categories.values())}")
        print(f"  Mapeo binario: {train_dataset.category_mapping}")
        
        # Analizar distribución de clases
        print(f"\nVerificando muestra del dataset...")
        class_distribution = {0: 0, 1: 0}
        
        # Revisar primeras 10 imágenes o todas si hay menos
        sample_size = min(10, len(train_dataset))
        
        for i in range(sample_size):
            sample = train_dataset[i]
            cat_ids = sample['category_ids'].tolist()
            for cat_id in cat_ids:
                if cat_id in class_distribution:
                    class_distribution[cat_id] += 1
        
        print(f"  Muestras analizadas: {sample_size} imágenes")
        print(f"  Distribución de clases encontrada:")
        print(f"    - Clase 0 (no cataract): {class_distribution[0]} máscaras")
        print(f"    - Clase 1 (cataract): {class_distribution[1]} máscaras")
        
        # Verificación final
        print(f"\n{'='*80}")
        print("RESULTADO DE LA PRUEBA:")
        
        unique_classes = set()
        for i in range(sample_size):
            sample = train_dataset[i]
            unique_classes.update(sample['category_ids'].tolist())
        
        if unique_classes.issubset({0, 1}):
            print("✓ ¡ÉXITO! El mapeo binario funciona correctamente.")
            print(f"  Clases únicas encontradas: {sorted(unique_classes)}")
            print(f"  Todas las categorías han sido mapeadas a 0 o 1.")
        else:
            print(f"✗ ADVERTENCIA: Se encontraron clases inesperadas: {unique_classes}")
        
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: No se pudo encontrar el archivo o directorio.")
        print(f"  Mensaje: {e}")
        print(f"\n  Asegúrate de:")
        print(f"    1. Tener el dataset COCO en la ruta correcta")
        print(f"    2. Actualizar la ruta en Core Modules/config.py si es necesario")
        
    except Exception as e:
        print(f"\n✗ Error inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_binary_mapping()
