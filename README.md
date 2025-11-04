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
### Opción 1: Quick Start (Recomendado)

Desde cualquier ubicación dentro del proyecto:

```bash
# Desde la raíz del proyecto
python "Utility Scripts/quickstart.py"

# O desde la carpeta Utility Scripts
cd "Utility Scripts"
python quickstart.py
```

### Opción 2: Scripts Directos

```bash
# Entrenamiento
python "Main Scripts/train.py" --checkpoint checkpoints/sam_vit_b_01ec64.pth

# Optimización con Optuna
python "Main Scripts/optuna_tuning.py" --checkpoint checkpoints/sam_vit_b_01ec64.pth --n_trials 50

# Inferencia
python "Main Scripts/inference.py" --checkpoint outputs/best_model.pt --split test
```

### Opción 3: PowerShell Script (Windows)

```powershell
cd "Utility Scripts"
.\run_training.ps1
```

### Opción 4: Google Colab

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

## 🔧 Instalación

### Windows (PowerShell)

```powershell
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno
.\venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Crear carpeta para checkpoints
New-Item -ItemType Directory -Path "checkpoints" -Force

# 5. Descargar checkpoint de SAM
# Opción A: Usar Invoke-WebRequest (PowerShell)
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "checkpoints/sam_vit_b_01ec64.pth"

# Opción B: Descargar manualmente desde:
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# y guardar en: checkpoints/

# 6. Verificar instalación
python "Utility Scripts/verify_setup.py"
```

### Linux/macOS (Bash)

```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar checkpoint de SAM
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b_01ec64.pth

# 5. Verificar instalación
python "Utility Scripts/verify_setup.py"
```

## 📊 Ventajas de la Estructura del Proyecto

1. **Organización Clara**: Separación lógica entre módulos core, scripts principales y utilidades
2. **Fácil Navegación**: Estructura intuitiva tipo IDE/proyecto profesional
3. **Modularidad**: Core Modules puede ser importado como paquete independiente
4. **Escalabilidad**: Fácil agregar nuevos scripts o módulos
5. **Mantenibilidad**: Código relacionado agrupado lógicamente
6. **Profesional**: Estructura estándar de proyectos Python
7. **Multiplataforma**: Compatible con Windows, Linux y macOS
8. **Colab-Ready**: Incluye soporte completo para Google Colab

## 🎯 Flujos de Trabajo Comunes

### Primera Vez

```bash
# 1. Verificar todo
python "Utility Scripts/verify_setup.py"

# 2. Empezar con quickstart
python "Utility Scripts/quickstart.py"
```

### Entrenamiento Estándar

**Bash/Linux/macOS:**
```bash
python "Main Scripts/train.py" \
    --checkpoint checkpoints/sam_vit_b_01ec64.pth \
    --batch_size 4 \
    --num_epochs 100
```

**PowerShell/Windows:**
```powershell
python "Main Scripts/train.py" `
    --checkpoint checkpoints/sam_vit_b_01ec64.pth `
    --batch_size 4 `
    --num_epochs 100
```

### Optimización de Hiperparámetros

**Bash/Linux/macOS:**
```bash
python "Main Scripts/optuna_tuning.py" \
    --checkpoint checkpoints/sam_vit_b_01ec64.pth \
    --n_trials 50
```

**PowerShell/Windows:**
```powershell
python "Main Scripts/optuna_tuning.py" `
    --checkpoint checkpoints/sam_vit_b_01ec64.pth `
    --n_trials 50
```

### Testing

**Bash/Linux/macOS:**
```bash
python "Main Scripts/inference.py" \
    --checkpoint outputs/sam_lora_cataract/checkpoints/best_model.pt \
    --split test \
    --save_visualizations
```

**PowerShell/Windows:**
```powershell
python "Main Scripts/inference.py" `
    --checkpoint outputs/sam_lora_cataract/checkpoints/best_model.pt `
    --split test `
    --save_visualizations
```

### Uso en Google Colab

**Opción A: Preparar paquete localmente**

```bash
# Preparar paquete para Colab
python "Utility Scripts/prepare_for_colab.py"
# Esto genera un archivo ZIP que puedes subir a Colab
```

**Opción B: Usar notebook directamente**

1. Subir `SAM_LoRA_Fine_tuning_colab_setup.ipynb` a Google Colab
2. Seguir las instrucciones en el notebook
3. El notebook incluye todo lo necesario para entrenar

## 📚 Recursos Adicionales

Para más información y casos de uso avanzados:

- `advanced_examples.py` - Ejemplos de uso avanzado del framework
- `SAM_LoRA_Fine_tuning_colab_setup.ipynb` - Notebook completo para Google Colab
- `Utility Scripts/prepare_for_colab.py` - Script para preparar paquete Colab
- Documentación de código en cada módulo con docstrings detallados

## ✅ Checklist de Instalación

Antes de comenzar a entrenar, verifica que todo esté configurado:

- [ ] Entorno virtual creado y activado
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

## 🐛 Troubleshooting

### Error: "No module named 'Core Modules'"

Los scripts automáticamente agregan `Core Modules` al path. Si encuentras este error:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Core Modules"))
```

### Error: CUDA out of memory

Reduce el `batch_size`:

```bash
# Linux/macOS/Windows
python "Main Scripts/train.py" --batch_size 2
```

### Error: Checkpoint no encontrado

Asegúrate de haber descargado el checkpoint SAM:

**PowerShell (Windows):**
```powershell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "checkpoints/sam_vit_b_01ec64.pth"
```

**Bash (Linux/macOS):**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b_01ec64.pth
```

## 📄 Licencia

Este proyecto utiliza SAM (Segment Anything Model) de Meta AI.

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para mejoras o correcciones.

---

¡Ya estás listo para usar el proyecto con su estructura organizada! 🚀
