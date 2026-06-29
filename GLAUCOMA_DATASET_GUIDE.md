# Guía del Dataset de Glaucoma (Todo Dataset)

## Descripción del Dataset

El `Todo Dataset/` es un dataset de fondo de ojo (fundus photography) para el análisis de glaucoma. Contiene imágenes de nervios ópticos etiquetadas por médicos como:

- **Normal**: Ojo sano sin signos de glaucoma
- **Pathological**: Ojo con signos de glaucoma

## Estructura del Dataset

```
Todo Dataset/
├── {id}_{left|right}.jpg          # Imagen original del fondo de ojo
├── {id}_{left|right}_cup.npy      # Máscara binaria del optic cup (excavación)
├── {id}_{left|right}_cup.png      # Máscara del optic cup en PNG
├── {id}_{left|right}_disc.npy     # Máscara binaria del optic disc (disco óptico)
├── {id}_{left|right}_disc.png     # Máscara del optic disc en PNG
├── annotations.json               # Metadatos y etiquetas de todas las imágenes
├── split.json                     # División train/test/validation
└── Organizacion del Dataset.png   # Distribución del split
```

## Estadísticas del Dataset

- **Total de imágenes**: 129
  - **Normal**: 60
  - **Pathological**: 69
- **Imágenes con máscaras de segmentación**: 69 (solo las Pathological)
- **Imágenes sin máscaras**: 60 (las Normal)

### Distribución del Split

| Partición | Total | %     | Normal | Glaucoma |
|-----------|-------|-------|--------|----------|
| Train     | 77    | 59.7% | 36     | 41       |
| Test      | 26    | 20.2% | 12     | 14       |
| Validation| 26    | 20.2% | 12     | 14       |
| **Total** | **129**| **100%** | **60** | **69** |

Todas las imágenes patológicas tienen tanto máscara de `cup` como de `disc`. Sin embargo, para este experimento se usa únicamente la máscara de `disc`.

### Imágenes con máscaras por split

| Partición | Con máscaras | Sin máscaras |
|-----------|--------------|--------------|
| Train     | 41           | 36           |
| Validation| 14           | 12           |
| Test      | 14           | 12           |

## Estrategia de Entrenamiento

El modelo se entrena con **TODAS las imágenes del split correspondiente**, tanto normales como patológicas:

- **Imágenes patológicas (Pathological)**: se usa la máscara real del disco óptico (`{id}_{side}_disc.npy`)
- **Imágenes normales (Normal)**: se usa una **máscara vacía** (todos los píxeles en 0)

Esto permite que el modelo vea ambos tipos de ojos durante el entrenamiento y aprenda a segmentar el disco cuando está presente y a no segmentar nada en imágenes normales.

> **Nota importante**: Anatómicamente, los ojos normales también tienen disco óptico. Sin embargo, como no está anotado en este dataset, se entrena con máscara vacía como estrategia para trabajar con los datos disponibles.

## Clases de Segmentación

Por defecto, el modelo se entrena para segmentar únicamente el **optic disc** (disco óptico):

| Clase | Nombre       | Descripción                          |
|-------|--------------|--------------------------------------|
| 0     | `optic_disc` | Disco óptico completo                |

> **Nota**: Aunque el dataset incluye máscaras tanto de `cup` como de `disc`, la configuración por defecto usa solo `disc`. Si en el futuro deseas entrenar también con `cup`, puedes modificar `mask_suffixes` en `GlaucomaDataConfig` o pasar `--mask_suffixes cup disc`.

## Cambios Realizados en el Código

Para soportar este dataset se implementaron las siguientes modificaciones:

### 1. `Core Modules/config.py`
- Agregado `dataset_type` en `DataConfig` para seleccionar entre `'coco'` y `'glaucoma'`
- Nueva clase `GlaucomaDataConfig` con la configuración específica del dataset
- **Por defecto usa solo la máscara de `disc`**

### 2. `Core Modules/dataset.py`
- Nueva clase `GlaucomaDataset` que carga:
  - `split.json` para la división train/val/test
  - Máscaras `.npy` (por defecto solo `disc`)
  - Genera automáticamente bounding boxes y point prompts
- Función `create_glaucoma_dataloaders()`
- `create_dataloaders()` ahora selecciona automáticamente el dataset según `config.data.dataset_type`

### 3. `Main Scripts/train.py`
- Nuevos argumentos `--dataset_type` y `--glaucoma_root`
- Configuración automática del dataset seleccionado

### 4. `Main Scripts/inference.py`
- Nuevos argumentos `--dataset_type` y `--glaucoma_root`
- Soporte para evaluar en el dataset de glaucoma

### 5. `Utility Scripts/verify_setup.py`
- Nueva función `check_glaucoma_data()`
- Argumento `--dataset_type glaucoma` para verificar este dataset

### 6. `test_glaucoma_dataset.py`
- Script de prueba para verificar la carga del dataset sin ejecutar entrenamiento

### 7. `requirements.txt`
- Fijado `numpy<2.0.0` para evitar incompatibilidades con albumentations, transformers, peft y tensorflow

## Pasos para Entrenar

### 0. Preparar el entorno

```bash
# Crear y activar entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

> **Importante**: Si ya tenías instalado `numpy>=2`, ejecuta:
> ```bash
> pip install "numpy<2" --force-reinstall
> ```

### 1. Descargar checkpoint de SAM

```bash
mkdir checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b_01ec64.pth
```

En Windows PowerShell:
```powershell
mkdir checkpoints
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "checkpoints/sam_vit_b_01ec64.pth"
```

### 2. Verificar el dataset

```bash
python "Utility Scripts/verify_setup.py" --dataset_type glaucoma
```

O usar el script de prueba:
```bash
python test_glaucoma_dataset.py
```

### 3. Entrenar con LoRA

```bash
python "Main Scripts/train.py" \
    --dataset_type glaucoma \
    --glaucoma_root "Todo Dataset" \
    --checkpoint checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --lora_rank 8 \
    --lora_alpha 16 \
    --batch_size 2 \
    --num_epochs 100 \
    --lr 1e-4 \
    --experiment_name sam_lora_glaucoma \
    --output_dir ./outputs
```

> **Nota**: Con solo 41 imágenes de entrenamiento, considera reducir `--num_epochs` y usar `--patience` menor para early stopping.

### 4. Inferencia

```bash
python "Main Scripts/inference.py" \
    --dataset_type glaucoma \
    --glaucoma_root "Todo Dataset" \
    --checkpoint outputs/sam_lora_glaucoma/checkpoints/best_model.pt \
    --split test \
    --save_visualizations \
    --output_dir ./inference_results_glaucoma
```

## Consideraciones Importantes

1. **Imágenes normales con máscaras vacías**: Las 60 imágenes normales no tienen anotaciones de segmentación. Se incluyen en el entrenamiento con máscaras vacías para que el modelo vea ambas clases de ojos.

2. **Dataset pequeño**: Con solo 41 imágenes de entrenamiento, el fine-tuning con LoRA es factible pero los resultados pueden ser limitados. Considera:
   - Data augmentation más agresivo
   - Cross-validation
   - Transfer learning desde modelos pre-entrenados en datasets similares

3. **Objetos pequeños**: El disco óptico es relativamente pequeño en la imagen completa. El resize a 1024x1024 puede perder detalles. Para futuras mejoras considera:
   - Crop alrededor de la región del disco óptico
   - Tamaños de imagen mayores
   - Prompts más precisos

4. **Hardware**: Se recomienda GPU para entrenar SAM. El entrenamiento en CPU será muy lento.

## Solución de Problemas

### Error con NumPy 2.x
Si ves errores como:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Ejecuta:
```bash
pip install "numpy<2" --force-reinstall
```

### CUDA Out of Memory
Reduce el batch size:
```bash
python "Main Scripts/train.py" --dataset_type glaucoma --batch_size 1
```

### Imágenes sin máscaras
Esto es normal. El script automáticamente ignora las imágenes normales (sin máscaras) durante el entrenamiento de segmentación.
