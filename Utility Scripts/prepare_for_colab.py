"""
Script para preparar el proyecto SAM LoRA para ser usado en Google Colab.
Crea un archivo ZIP optimizado y genera instrucciones de uso.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

# Agregar Core Modules al path
sys.path.insert(0, str(Path(__file__).parent.parent / "Core Modules"))


def create_colab_package(output_dir: str = None) -> str:
    """
    Crea un paquete ZIP del proyecto optimizado para Google Colab.
    
    Args:
        output_dir: Directorio donde guardar el ZIP. Si es None, usa el directorio padre.
    
    Returns:
        Ruta al archivo ZIP creado.
    """
    project_root = Path(__file__).parent.parent
    
    if output_dir is None:
        output_dir = project_root.parent
    else:
        output_dir = Path(output_dir)
    
    # Nombre del archivo ZIP con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"SAM_LoRA_Colab_{timestamp}.zip"
    zip_path = output_dir / zip_name
    
    print(f"📦 Creando paquete para Google Colab...")
    print(f"   Directorio del proyecto: {project_root}")
    print(f"   Archivo de salida: {zip_path}")
    
    # Archivos y carpetas a incluir
    include_items = [
        "Core Modules",
        "Main Scripts",
        "Utility Scripts",
        "Documentation",
        "requirements.txt",
        ".gitignore",
        "advanced_examples.py",
        "colab_setup.ipynb",
        "ESTRUCTURA_README.md"
    ]
    
    # Archivos a excluir (patterns)
    exclude_patterns = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".git",
        ".vscode",
        ".idea",
        "outputs",
        "checkpoints/*.pth",  # No incluir checkpoints grandes
        "*.pt",
        "*.pth",
        ".DS_Store",
        "*.log",
        "wandb",
        "optuna_study.db"
    ]
    
    def should_exclude(file_path: Path) -> bool:
        """Verifica si un archivo debe ser excluido."""
        path_str = str(file_path)
        for pattern in exclude_patterns:
            if pattern in path_str or file_path.name == pattern:
                return True
            if pattern.startswith("*.") and file_path.name.endswith(pattern[1:]):
                return True
        return False
    
    # Crear archivo ZIP
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in include_items:
            item_path = project_root / item
            
            if not item_path.exists():
                print(f"⚠️  Omitiendo {item} (no existe)")
                continue
            
            if item_path.is_file():
                if not should_exclude(item_path):
                    arcname = f"SAM finetuning LoRA/{item}"
                    zipf.write(item_path, arcname)
                    print(f"   ✓ {item}")
            
            elif item_path.is_dir():
                for root, dirs, files in os.walk(item_path):
                    # Filtrar directorios a excluir
                    dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d)]
                    
                    for file in files:
                        file_path = Path(root) / file
                        
                        if not should_exclude(file_path):
                            # Mantener estructura de carpetas
                            rel_path = file_path.relative_to(project_root)
                            arcname = f"SAM finetuning LoRA/{rel_path}"
                            zipf.write(file_path, arcname)
                
                print(f"   ✓ {item}/")
    
    # Obtener tamaño del archivo
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Paquete creado exitosamente!")
    print(f"   Tamaño: {size_mb:.2f} MB")
    print(f"   Ubicación: {zip_path}")
    
    return str(zip_path)


def generate_colab_instructions(zip_path: str) -> None:
    """
    Genera instrucciones de uso para Google Colab.
    
    Args:
        zip_path: Ruta al archivo ZIP creado.
    """
    instructions = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    INSTRUCCIONES PARA GOOGLE COLAB                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

📦 Archivo creado: {Path(zip_path).name}

🚀 PASOS PARA USAR EN GOOGLE COLAB:

1️⃣  SUBIR A GOOGLE DRIVE:
    a) Abre Google Drive (drive.google.com)
    b) Crea una carpeta llamada "SAM_LoRA_Training"
    c) Sube el archivo {Path(zip_path).name} a esa carpeta

2️⃣  ABRIR NOTEBOOK EN COLAB:
    a) En Google Drive, haz doble clic en "colab_setup.ipynb" (dentro del ZIP)
    b) O ve a colab.research.google.com
    c) File → Upload notebook → Selecciona colab_setup.ipynb

3️⃣  CONFIGURAR GPU:
    Runtime → Change runtime type → Hardware accelerator → GPU (T4)

4️⃣  EJECUTAR CELDAS DEL NOTEBOOK:
    Sigue las instrucciones en cada celda del notebook.
    El notebook te guiará paso a paso.

📚 ARCHIVOS IMPORTANTES:

    • colab_setup.ipynb                 → Notebook principal para Colab
    • Documentation/GOOGLE_COLAB_GUIDE.md → Guía detallada completa
    • ESTRUCTURA_README.md              → Documentación de la estructura
    • requirements.txt                  → Dependencias de Python

⚡ COMANDOS RÁPIDOS PARA COLAB:

    # Montar Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Descomprimir proyecto
    !unzip -q "/content/drive/MyDrive/SAM_LoRA_Training/{Path(zip_path).name}" -d /content/
    
    # Entrar al directorio
    %cd "/content/SAM finetuning LoRA"
    
    # Descargar checkpoint SAM
    !mkdir -p checkpoints
    !wget -O checkpoints/sam_vit_b_01ec64.pth \\
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    
    # Entrenar (ejemplo rápido)
    !python "Main Scripts/train.py" \\
        --checkpoint checkpoints/sam_vit_b_01ec64.pth \\
        --batch_size 2 --num_epochs 5

🔍 TIPS:

    ✓ Lee Documentation/GOOGLE_COLAB_GUIDE.md para guía completa
    ✓ Usa el notebook colab_setup.ipynb para configuración guiada
    ✓ Guarda checkpoints en Drive para no perderlos
    ✓ Usa --mixed_precision para ahorrar memoria GPU
    ✓ Configura --save_frequency para checkpoints frecuentes

💡 SOLUCIÓN DE PROBLEMAS:

    • "CUDA Out of Memory"     → Reducir --batch_size a 1 o 2
    • "Runtime disconnected"   → Usar script anti-desconexión (ver guía)
    • "Import error"           → Verificar que estés en el directorio correcto
    • Dataset no encontrado    → Usar rutas absolutas a /content/

📖 MÁS INFORMACIÓN:

    Documentación completa: Documentation/GOOGLE_COLAB_GUIDE.md
    Estructura del proyecto: ESTRUCTURA_README.md
    README principal: Documentation/README.md

╔══════════════════════════════════════════════════════════════════════════════╗
║  ¡Ya estás listo para entrenar SAM con LoRA en Google Colab! 🚀            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    print(instructions)
    
    # Guardar instrucciones en archivo
    instructions_path = Path(zip_path).parent / "COLAB_INSTRUCTIONS.txt"
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"📄 Instrucciones guardadas en: {instructions_path}")


def check_dataset_size(dataset_path: str = None) -> None:
    """
    Verifica el tamaño del dataset y da recomendaciones.
    
    Args:
        dataset_path: Ruta al dataset COCO. Si es None, usa la ruta por defecto.
    """
    if dataset_path is None:
        # Ruta por defecto relativa al proyecto
        project_root = Path(__file__).parent.parent
        dataset_path = project_root.parent / "Cataract COCO Segmentation" / "Cataract COCO Segmentation"
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"⚠️  Dataset no encontrado en: {dataset_path}")
        print("   Por favor especifica la ruta correcta con --dataset_path")
        return
    
    # Calcular tamaño total
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = Path(root) / file
            total_size += file_path.stat().st_size
            file_count += 1
    
    size_mb = total_size / (1024 * 1024)
    size_gb = size_mb / 1024
    
    print(f"\n📊 INFORMACIÓN DEL DATASET:")
    print(f"   Ubicación: {dataset_path}")
    print(f"   Archivos: {file_count:,}")
    print(f"   Tamaño total: {size_mb:.2f} MB ({size_gb:.2f} GB)")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES PARA COLAB:")
    
    if size_gb < 1:
        print("   ✓ Dataset pequeño (<1GB)")
        print("   → Puedes subirlo directamente a Colab")
        print("   → O guardar en Google Drive")
    elif size_gb < 5:
        print("   ⚠️  Dataset mediano (1-5GB)")
        print("   → Mejor guardar en Google Drive")
        print("   → Comprimir antes de subir (usa .zip)")
    else:
        print("   🔴 Dataset grande (>5GB)")
        print("   → DEBE estar en Google Drive")
        print("   → Considerar comprimir o reducir tamaño de imágenes")
        print("   → Alternativa: usar Roboflow para descarga directa en Colab")
    
    # Crear ZIP del dataset si es pequeño
    if size_gb < 1:
        print(f"\n❓ ¿Crear ZIP del dataset para Colab?")
        response = input("   (s/n): ").lower().strip()
        
        if response == 's':
            output_dir = dataset_path.parent
            zip_name = "cataract_dataset_colab.zip"
            zip_path = output_dir / zip_name
            
            print(f"\n📦 Comprimiendo dataset...")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(dataset_path):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(dataset_path.parent)
                        zipf.write(file_path, arcname)
            
            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ Dataset comprimido: {zip_path}")
            print(f"   Tamaño: {zip_size_mb:.2f} MB")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepara el proyecto SAM LoRA para Google Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
    
    # Crear paquete básico
    python prepare_for_colab.py
    
    # Especificar directorio de salida
    python prepare_for_colab.py --output_dir "C:/Users/Usuario/Desktop"
    
    # Incluir análisis del dataset
    python prepare_for_colab.py --check_dataset
    
    # Especificar ruta del dataset
    python prepare_for_colab.py --check_dataset --dataset_path "ruta/al/dataset"
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio donde guardar el ZIP (default: directorio padre del proyecto)'
    )
    
    parser.add_argument(
        '--check_dataset',
        action='store_true',
        help='Verificar tamaño del dataset y dar recomendaciones'
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Ruta al dataset COCO (default: ../Cataract COCO Segmentation/...)'
    )
    
    args = parser.parse_args()
    
    try:
        # Crear paquete ZIP
        zip_path = create_colab_package(args.output_dir)
        
        # Generar instrucciones
        generate_colab_instructions(zip_path)
        
        # Verificar dataset si se solicita
        if args.check_dataset:
            check_dataset_size(args.dataset_path)
        
        print("\n" + "="*80)
        print("✅ PREPARACIÓN COMPLETADA")
        print("="*80)
        print(f"\n📦 Sube estos archivos a Google Drive:")
        print(f"   1. {Path(zip_path).name}")
        if args.check_dataset:
            print(f"   2. cataract_dataset_colab.zip (si se creó)")
        print(f"\n📖 Lee COLAB_INSTRUCTIONS.txt para más detalles")
        print(f"\n🚀 Luego abre colab_setup.ipynb en Google Colab")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
