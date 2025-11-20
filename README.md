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

Este proyecto utiliza SAM (Segment Anything Model) de Meta AI.

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para mejoras o correcciones.

---
