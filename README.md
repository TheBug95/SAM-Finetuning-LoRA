# SAM LoRA Fine-tuning

## 📁 Estructura del Proyecto

```text
SAM finetuning LoRA/
│
├── Core Modules/              # Módulos principales del framework
│   ├── __init__.py           # Inicialización del paquete
│   ├── config.py             # Configuración del proyecto
│   ├── dataset.py            # Carga y procesamiento de datos COCO
│   ├── model.py              # Modelo SAM con LoRA
│   ├── trainer.py            # Lógica de entrenamiento
│   └── utils.py              # Utilidades (losses, métricas, etc.)
│
├── Main Scripts/              # Scripts principales de ejecución
│   ├── train.py              # Script de entrenamiento
│   ├── inference.py          # Script de inferencia/testing
│   └── optuna_tuning.py      # Optimización de hiperparámetros
│
├── Utility Scripts/           # Scripts auxiliares
│   ├── quickstart.py         # Interfaz interactiva para comenzar
│   ├── verify_setup.py       # Verificación de instalación
│   ├── export_to_huggingface.py  # Exportar a HuggingFace Hub
│   ├── prepare_for_colab.py  # Preparar paquete para Google Colab
│   └── run_training.ps1      # Script PowerShell con menú
│
├── SAM_LoRA_Fine_tuning_colab_setup.ipynb  # Notebook para Google Colab
├── requirements.txt           # Dependencias de Python
├── advanced_examples.py       # Ejemplos de uso avanzado
└── .gitignore                # Archivos a ignorar en Git
```

## 🚀 Cómo Usar 

### Google Colab

Para usar el proyecto en Google Colab:

```bash
# Usar directamente el notebook incluido
# Subir SAM_LoRA_Fine_tuning_colab_setup.ipynb a Google Colab
```


## 📝 Notas Importantes

### 1. Imports Automáticos

Los scripts en `Main Scripts` y `Utility Scripts` automáticamente agregan `Core Modules` al path de Python, por lo que no necesitas preocuparte por los imports.

### 2. Rutas Relativas

Todos los scripts usan rutas relativas correctas:

- Los scripts buscan archivos relativos al directorio raíz del proyecto
- Los checkpoints se guardan en `outputs/` desde la raíz
- Los datos se buscan en la ruta especificada en `--data_root`

### 3. Verificación del Setup

Antes de empezar, ejecuta:

```bash
python "Utility Scripts/verify_setup.py"
```

Este script verifica:

- ✓ Versión de Python
- ✓ PyTorch y CUDA
- ✓ Dependencias instaladas
- ✓ Checkpoint de SAM
- ✓ Datos COCO
- ✓ Estructura de archivos
- ✓ Imports de módulos

### Uso en Google Colab

** Usar notebook directamente**

1. Subir `SAM_LoRA_Fine_tuning_colab_setup.ipynb` a Google Colab
2. Seguir las instrucciones en el notebook
3. El notebook incluye todo lo necesario para entrenar

## ✅ Checklist de Instalación

Antes de comenzar a entrenar, verifica que todo esté configurado:

- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Checkpoint de SAM descargado en `checkpoints/sam_vit_b_01ec64.pth`
- [ ] Dataset COCO preparado y accesible
- [ ] `verify_setup.py` ejecutado exitosamente sin errores
- [ ] CUDA disponible (opcional, pero recomendado para entrenamiento)

## 🎓 Scripts Disponibles

### Scripts Principales (`Main Scripts/`)

- **`train.py`**: Script principal de entrenamiento con LoRA
- **`inference.py`**: Evaluación y testing del modelo entrenado
- **`optuna_tuning.py`**: Optimización automática de hiperparámetros

### Scripts Utilitarios (`Utility Scripts/`)

- **`quickstart.py`**: Interfaz interactiva para comenzar rápidamente
- **`verify_setup.py`**: Verificación completa del entorno
- **`export_to_huggingface.py`**: Exportar modelo a HuggingFace Hub
- **`prepare_for_colab.py`**: Preparar paquete optimizado para Google Colab
- **`run_training.ps1`**: Script PowerShell con menú interactivo (Windows)

## 🔍 Inferencia

El script `Main Scripts/inference.py` puede ejecutarse en tres modalidades:

### 1. SAM base (sin entrenar)

Usa los pesos preentrenados oficiales de SAM. No es necesario pasar `--checkpoint`, pero sí debes descargar el checkpoint oficial correspondiente a `--model_type`.

```bash
python "Main Scripts/inference.py" \
    --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
    --data_root "Cataract COCO Segmentation/Cataract COCO Segmentation" \
    --split test \
    --output_dir ./resultados_sam_base
```

### 2. Modelo entrenado con LoRA

Carga un checkpoint propio generado durante el entrenamiento. Si el checkpoint solo contiene los pesos LoRA, también puedes pasar `--sam_checkpoint` para cargar los pesos base de SAM.

```bash
python "Main Scripts/inference.py" \
    --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
    --checkpoint outputs/sam_lora_cataract/checkpoints/best_model.pt \
    --data_root "Cataract COCO Segmentation/Cataract COCO Segmentation" \
    --split test \
    --output_dir ./resultados_lora \
    --save_visualizations
```

### 3. Checkpoint autocontenido

Si tu checkpoint de entrenamiento ya incluye los pesos base de SAM, basta con:

```bash
python "Main Scripts/inference.py" \
    --checkpoint outputs/sam_lora_cataract/checkpoints/best_model.pt \
    --data_root "Cataract COCO Segmentation/Cataract COCO Segmentation" \
    --split test \
    --output_dir ./resultados_lora
```

### Máscaras generadas

Por defecto, el script guarda todas las máscaras predichas por SAM en la subcarpeta `sam_masks` dentro de `--output_dir`. Cada máscara se nombra con el nombre de la imagen de origen:

```text
resultados_lora/
├── sam_masks/
│   ├── imagen_001_obj_0.png
│   ├── imagen_001_obj_1.png
│   ├── imagen_002_obj_0.png
│   └── ...
├── visualizations/
├── metrics.json
└── detailed_results.json
```

Puedes cambiar el nombre de la carpeta con `--masks_dir`.

## 🔬 Dataset de Glaucoma (Todo Dataset)

Este repositorio ahora también soporta el dataset de glaucoma `Todo Dataset/` para segmentación del **optic disc** (disco óptico) en imágenes de fondo de ojo. Por defecto se entrena únicamente con la máscara del disco; la máscara del cup está disponible pero no se usa.

Para más detalles, consulta la guía completa:
- **[GLAUCOMA_DATASET_GUIDE.md](GLAUCOMA_DATASET_GUIDE.md)**

Comandos rápidos:

```bash
# Verificar dataset
python "Utility Scripts/verify_setup.py" --dataset_type glaucoma

# Entrenar
python "Main Scripts/train.py" \
    --dataset_type glaucoma \
    --glaucoma_root "Todo Dataset" \
    --checkpoint checkpoints/sam_vit_b_01ec64.pth \
    --batch_size 2 \
    --experiment_name sam_lora_glaucoma

# Inferir
python "Main Scripts/inference.py" \
    --dataset_type glaucoma \
    --glaucoma_root "Todo Dataset" \
    --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
    --checkpoint outputs/sam_lora_glaucoma/checkpoints/best_model.pt \
    --split test \
    --save_visualizations
```

Este proyecto utiliza SAM (Segment Anything Model) de Meta AI.

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para mejoras o correcciones.

---
