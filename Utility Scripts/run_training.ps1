# Script conveniente para entrenar con configuraciones pre-definidas
# Para Windows PowerShell

# Get script directory and move to parent (project root)
$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

# Configuración básica
$CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"
$DATA_ROOT = "../Cataract COCO Segmentation/Cataract COCO Segmentation"
$OUTPUT_DIR = "./outputs"
$TRAIN_SCRIPT = "Main Scripts/train.py"
$OPTUNA_SCRIPT = "Main Scripts/optuna_tuning.py"
$INFERENCE_SCRIPT = "Main Scripts/inference.py"

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "="*59 -ForegroundColor Cyan
Write-Host "SAM LoRA Training Scripts" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "="*59 -ForegroundColor Cyan

Write-Host "`nSelecciona una opción:" -ForegroundColor Yellow
Write-Host "1. Entrenamiento básico (batch_size=4, epochs=100)"
Write-Host "2. Entrenamiento rápido (batch_size=8, epochs=50)"
Write-Host "3. Entrenamiento largo (batch_size=2, epochs=200)"
Write-Host "4. Optimización con Optuna (50 trials)"
Write-Host "5. Inferencia en test set"
Write-Host "6. Salir"

$choice = Read-Host "`nOpción"

switch ($choice) {
    "1" {
        Write-Host "`n🚀 Iniciando entrenamiento básico..." -ForegroundColor Green
        python $TRAIN_SCRIPT `
            --checkpoint $CHECKPOINT `
            --data_root $DATA_ROOT `
            --experiment_name sam_lora_basic `
            --batch_size 4 `
            --num_epochs 100 `
            --lr 1e-4 `
            --lora_rank 8 `
            --lora_alpha 16 `
            --output_dir $OUTPUT_DIR
    }
    
    "2" {
        Write-Host "`n⚡ Iniciando entrenamiento rápido..." -ForegroundColor Green
        python $TRAIN_SCRIPT `
            --checkpoint $CHECKPOINT `
            --data_root $DATA_ROOT `
            --experiment_name sam_lora_fast `
            --batch_size 8 `
            --num_epochs 50 `
            --lr 2e-4 `
            --lora_rank 8 `
            --lora_alpha 16 `
            --output_dir $OUTPUT_DIR
    }
    
    "3" {
        Write-Host "`n🐢 Iniciando entrenamiento largo..." -ForegroundColor Green
        python $TRAIN_SCRIPT `
            --checkpoint $CHECKPOINT `
            --data_root $DATA_ROOT `
            --experiment_name sam_lora_long `
            --batch_size 2 `
            --num_epochs 200 `
            --lr 5e-5 `
            --lora_rank 16 `
            --lora_alpha 32 `
            --output_dir $OUTPUT_DIR
    }
    
    "4" {
        Write-Host "`n🔍 Iniciando optimización con Optuna..." -ForegroundColor Green
        python $OPTUNA_SCRIPT `
            --checkpoint $CHECKPOINT `
            --data_root $DATA_ROOT `
            --n_trials 50 `
            --study_name sam_lora_optimization `
            --storage "sqlite:///optuna_study.db" `
            --output_dir $OUTPUT_DIR
    }
    
    "5" {
        Write-Host "`n🧪 Iniciando inferencia..." -ForegroundColor Green
        
        # Buscar el último checkpoint
        $checkpoints = Get-ChildItem -Path "$OUTPUT_DIR/*/checkpoints/best_model.pt" -Recurse -ErrorAction SilentlyContinue
        
        if ($checkpoints.Count -eq 0) {
            Write-Host "⚠️  No se encontraron modelos entrenados!" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "`nModelos disponibles:" -ForegroundColor Yellow
        for ($i = 0; $i -lt $checkpoints.Count; $i++) {
            Write-Host "$($i+1). $($checkpoints[$i].FullName)"
        }
        
        if ($checkpoints.Count -eq 1) {
            $modelPath = $checkpoints[0].FullName
        } else {
            $modelChoice = Read-Host "`nSelecciona modelo (1-$($checkpoints.Count))"
            $modelPath = $checkpoints[$modelChoice - 1].FullName
        }
        
        python $INFERENCE_SCRIPT `
            --checkpoint $modelPath `
            --data_root $DATA_ROOT `
            --split test `
            --output_dir "./inference_results" `
            --save_visualizations
    }
    
    "6" {
        Write-Host "`n👋 Hasta luego!" -ForegroundColor Green
        exit 0
    }
    
    default {
        Write-Host "`n⚠️  Opción inválida!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n✅ Proceso completado!" -ForegroundColor Green
