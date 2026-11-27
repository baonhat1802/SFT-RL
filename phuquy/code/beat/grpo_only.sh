#!/bin/bash
# Mốc GRPO-only trong pipeline của mình, cấu hình bảng nhóm: LoRA r32 α64, lr 1e-5, trần 512 token, 1 epoch curated.
#   bash code/beat/grpo_only.sh <seed> [smoke]
set -uo pipefail
cd ~/capstone
PY=~/miniconda3/envs/capstone/bin/python
export PYTHONPATH=~/capstone/code HF_HOME=~/hf TOKENIZERS_PARALLELISM=false CAPSTONE_REWARD_WORKERS=8
S=${1:-1}; MODE=${2:-full}; R=runs; EXTRA=""
if [ "$MODE" = smoke ]; then R=runs_smoke; EXTRA="--max-steps 2 --prompts-per-rollout 16"; fi
NAME=grpo_only_s$S
mkdir -p $R/logs
echo "=== $(date -u +%H:%M:%S) train $NAME"
$PY -m train.grpo --track curated --baseline grpo --seed $S --lora-r 32 --lora-alpha 64 --lr 1e-5 --max-completion 512 \
    --run-name $NAME --output-dir $R/$NAME $EXTRA > $R/logs/train_$NAME.log 2>&1 || { echo "FAIL train"; tail -40 $R/logs/train_$NAME.log; exit 1; }
grep -oE "'reward': [0-9.]+" $R/logs/train_$NAME.log | tail -3
echo "=== $(date -u +%H:%M:%S) eval $NAME"
$PY -m beat.eval_bridge --model Qwen/Qwen2.5-3B --adapter $R/$NAME/final_adapter --max-lora-rank 32 --max-tokens 512 --n 4 \
    --out $R/eval/$NAME > $R/logs/eval_$NAME.log 2>&1 || { echo "FAIL eval"; tail -30 $R/logs/eval_$NAME.log; }
grep RESULT $R/logs/eval_$NAME.log
echo "=== $(date -u +%H:%M:%S) XONG $NAME"
