#!/usr/bin/env bash
#
# ASE-MTAGE LunarLander-v2 formal experiment (nohup)
#
# Usage:  bash run_experiment.sh
# Logs:   tail -f outputs/ase_mtage_lunarlander_v3/run.log
# Stop:   kill $(cat outputs/ase_mtage_lunarlander_v3/pid.txt)
#
set -euo pipefail

cd /home/utseus22/eure

EXPERIMENT_NAME="ase_mtage_lunarlander_v3"
CONFIG="configs/ase_mtage_lunarlander_formal.json"
GPU_ID=1

export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_FILE="outputs/${EXPERIMENT_NAME}/run.log"
mkdir -p "outputs/${EXPERIMENT_NAME}"

echo "GPU:           $(nvidia-smi -i $GPU_ID --query-gpu=name --format=csv,noheader)"
echo "Experiment:    $EXPERIMENT_NAME"
echo "Config:        $CONFIG"
echo "Log:           $LOG_FILE"
echo ""

nohup python3 -c "
import sys
sys.path.insert(0, '.')
from ase_mtage.pipeline import run_phase1

result = run_phase1(
    config_path='$CONFIG',
    experiment_name='$EXPERIMENT_NAME',
    n_rounds=10,
)
print('DONE:', result.get('success'), result.get('exp_dir'))
" > /dev/null 2>&1 &

PID=$!
echo $PID > "outputs/${EXPERIMENT_NAME}/pid.txt" 2>/dev/null || true
echo "Launched PID=$PID"
echo ""
echo "Monitor:  tail -f $LOG_FILE"
echo "GIFs:     outputs/${EXPERIMENT_NAME}/round*/full_training/eval_gifs/"
echo "Stop:     kill $PID"
