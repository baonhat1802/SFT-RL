#!/bin/bash
# Khói vòng R01 trên beat/loop.py: mỗi idea 2 rollout x 16 câu, rồi kiểm không có Traceback.
CAP_DIR=${CAP_DIR:-~/capstone}; cd $CAP_DIR
PY=${PY:-~/miniconda3/envs/capstone/bin/python}
export PYTHONPATH=$CAP_DIR/code HF_HOME=${HF_HOME:-~/hf} TOKENIZERS_PARALLELISM=false
mkdir -p runs_smoke/r01 runs_smoke/logs
for spec in "grpo" "guide_tokw" "fade" "compress_sd" "opsd" "metaopsd" "potential"; do
  echo "=== $(date -u +%H:%M:%S) smoke $spec"
  $PY -m beat.loop --idea $spec --seed 1 --limit 64 --prompts-per-rollout 16 --max-rollouts 2 \
      --out runs_smoke/r01/$spec > runs_smoke/logs/r01_$spec.log 2>&1 \
    && grep -E "^STEP" runs_smoke/logs/r01_$spec.log | tail -2 | cut -c1-300 \
    || { echo "FAIL $spec"; grep -B2 -A25 Traceback runs_smoke/logs/r01_$spec.log | tail -30; }
done
echo "=== XONG r01 smoke"
