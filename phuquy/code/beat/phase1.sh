#!/bin/bash
# Phase 1 trên VM GCP (1× A100 40GB): điểm neo đánh giá, sinh dữ liệu SFT kiểu "model tự viết", SFT, đánh giá.
#   bash code/beat/phase1.sh smoke     # chạy nhỏ để bắt lỗi (vài phút)
#   bash code/beat/phase1.sh all       # chạy thật
# Mỗi bước bỏ qua nếu đã có kết quả (summary.json / stats.json / final_adapter), nên chạy lại được sau khi hỏng.
set -uo pipefail
cd ~/capstone
PY=~/miniconda3/envs/capstone/bin/python
export PYTHONPATH=~/capstone/code HF_HOME=~/hf TOKENIZERS_PARALLELISM=false CAPSTONE_REWARD_WORKERS=10
MODE=${1:-all}
R=runs; LIM=""; SFTLIM=""; EVN=4
if [ "$MODE" = smoke ]; then R=runs_smoke; LIM="--limit 64"; SFTLIM="--max-steps 3"; EVN=2; fi
mkdir -p $R/logs
# SFT config của bảng nhóm: LoRA r32 alpha64, lr 1e-4, 3 epoch, batch 64
SFT_ARGS="--track curated --lora-r 32 --lora-alpha 64 --lr 1e-4 --per-device-batch 16 --no-merge"

step() { echo "=== $(date -u +%H:%M:%S) $*"; }

ev() {   # ev <name> [--adapter dir]
  local name=$1; shift
  [ -f $R/eval/$name/summary.json ] && { step "eval $name: có rồi"; return; }
  step "eval $name"
  $PY -m beat.eval_bridge --model Qwen/Qwen2.5-3B --max-tokens 512 --n $EVN --max-lora-rank 32 \
      --out $R/eval/$name "$@" > $R/logs/eval_$name.log 2>&1 || { echo "FAIL eval $name"; tail -30 $R/logs/eval_$name.log; }
  grep RESULT $R/logs/eval_$name.log
}

gen() {
  [ -f $R/data/$1/stats.json ] && { step "gen $1: có rồi"; return; }
  step "gen $1"
  $PY -m beat.gen_sft_data --mode $1 --out $R/data/$1 $LIM > $R/logs/gen_$1.log 2>&1 || { echo "FAIL gen $1"; tail -30 $R/logs/gen_$1.log; }
  grep STATS $R/logs/gen_$1.log
}

sft() {  # sft <name> [--train-file f]
  local name=$1; shift
  [ -d $R/sft_$name/final_adapter ] && { step "sft $name: có rồi"; return; }
  step "sft $name"
  $PY -m train.sft $SFT_ARGS --run-name sft_$name --output-dir $R/sft_$name $SFTLIM "$@" \
      > $R/logs/sft_$name.log 2>&1 || { echo "FAIL sft $name"; tail -30 $R/logs/sft_$name.log; }
  grep -o "'train_loss': [0-9.]*" $R/logs/sft_$name.log | tail -1
}

ev base512
gen hint
gen self
sft hint --train-file $R/data/hint/train.parquet && ev sft_hint --adapter $R/sft_hint/final_adapter
sft self --train-file $R/data/self/train.parquet && ev sft_self --adapter $R/sft_self/final_adapter
sft luna && ev sft_luna --adapter $R/sft_luna/final_adapter      # điểm neo: bảng nhóm báo 8.1
step "XONG $MODE"
