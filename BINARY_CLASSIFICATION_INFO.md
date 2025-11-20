# Clasificación Binaria de Cataratas

## Cambio en el Sistema de Clasificación

Este proyecto ahora utiliza **clasificación binaria** en lugar de las 4 clases originales del dataset.

### Mapeo de Categorías

El dataset COCO original contiene 4 categorías:
- `cataract` (catarata)
- `mild` (leve)
- `severe` (severa)
- `normal` (normal)

Estas han sido **remapeadas automáticamente** a 2 clases binarias:

| Clase Binaria | ID | Categorías Originales Incluidas | Descripción |
|---------------|----|---------------------------------|-------------|
| **No Cataract** | 0 | `normal` | Ojo sin catarata |
| **Cataract** | 1 | `cataract`, `mild`, `severe` | Ojo con catarata (cualquier severidad) |

## Implementación

### Cambios Realizados

1. **dataset.py** (`Core Modules/dataset.py`):
   - Agregado método `_create_binary_category_mapping()` que crea el mapeo automáticamente
   - Las categorías originales se convierten a clases binarias al cargar las anotaciones
   - Se mantienen compatibles con los archivos JSON de COCO sin modificarlos

2. **config.py** (`Core Modules/config.py`):
   - Agregado `num_classes = 2` en `DataConfig`
   - Agregado `class_names = ["no_cataract", "cataract"]`
   - Documentación clara sobre el mapeo binario

### Comportamiento del Código

Al cargar el dataset, verás un mensaje como este:

```
Loaded 150 images with annotations
Original categories: ['cataract', 'mild', 'normal', 'severe']
Binary mapping: {1: 1, 2: 1, 3: 0, 4: 1}
Classes: 0=no cataract, 1=cataract
```

Esto indica:
- Las categorías originales detectadas en el JSON
- El mapeo de IDs (por ejemplo: ID 1,2,4 → clase 1, ID 3 → clase 0)
- Las 2 clases finales utilizadas para entrenamiento

## Uso

No necesitas hacer ningún cambio adicional. El código:
- ✅ Lee automáticamente las 4 categorías del dataset COCO
- ✅ Las mapea a 2 clases binarias internamente
- ✅ Entrena el modelo con clasificación binaria
- ✅ No requiere modificar los archivos JSON originales

## Ventajas de la Clasificación Binaria

1. **Simplificación del Problema**: Reducir de 4 a 2 clases simplifica la tarea de segmentación
2. **Mayor Utilidad Clínica**: En muchos casos, solo es necesario detectar presencia/ausencia de catarata
3. **Datos Consolidados**: Agrupa todas las severidades de catarata en una sola clase
4. **Mejora en el Balance de Clases**: Potencialmente reduce problemas de desbalance

## Archivos Modificados

- `Core Modules/dataset.py`: Lógica de mapeo de categorías
- `Core Modules/config.py`: Configuración de clases
- Este archivo (`BINARY_CLASSIFICATION_INFO.md`): Documentación

## Notas Importantes

- Los archivos JSON de COCO **NO se modifican**
- El mapeo ocurre **en tiempo de carga** (runtime)
- Las máscaras y anotaciones originales se mantienen intactas
- Si necesitas volver a 4 clases, puedes comentar la línea de mapeo en `dataset.py`

## Verificación

Para verificar que el mapeo funciona correctamente, ejecuta:

```bash
python "Utility Scripts/verify_setup.py"
```

O carga un dataset y observa los mensajes de salida que muestran el mapeo aplicado.
