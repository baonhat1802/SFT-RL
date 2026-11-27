#!/bin/bash
# Chạy hàng đợi một vòng: mỗi dòng "<tên> <idea> [args...]" -> beat.loop rồi beat.eval_bridge @512 n=4.
# Nhiều worker (mỗi card một worker) dùng chung một hàng đợi: run được giành bằng mkdir <out>/<tên>.claim; run đã có summary bỏ qua.
#   bash code/beat/run_jobs.sh research/rounds/R01/jobs.txt runs/r01
# Biến môi trường: CAP_DIR (mặc định ~/capstone), PY, HF_HOME, CUDA_VISIBLE_DEVICES,
#   LOOP_ARGS (thêm vào mọi lệnh beat.loop, vd "--max-completion 2048 --micro-batch 4"), EVAL_TOKENS (mặc định "512").
CAP_DIR=${CAP_DIR:-~/capstone}; cd $CAP_DIR
PY=${PY:-~/miniconda3/envs/capstone/bin/python}
export PYTHONPATH=$CAP_DIR/code HF_HOME=${HF_HOME:-~/hf} TOKENIZERS_PARALLELISM=false
JOBS=$1; OUT=${2:-runs/r01}; W=${WORKER:-0}; mkdir -p $OUT/logs
grep -vE '^\s*(#|$)' $JOBS | while read -r name idea rest; do
  [ -f $OUT/$name/eval/summary.json ] && continue
  mkdir $OUT/$name.claim 2>/dev/null || continue
  echo "=== $(date -u +%H:%M:%S) [w$W] train $name ($idea $rest)"
  $PY -m beat.loop --idea $idea --seed 1 --out $OUT/$name ${LOOP_ARGS:-} $rest > $OUT/logs/train_$name.log 2>&1 \
    || { echo "FAIL train $name"; grep -A20 Traceback $OUT/logs/train_$name.log | tail -20; continue; }
  tail -1 $OUT/$name/log.jsonl | cut -c1-300
  for T in ${EVAL_TOKENS:-512}; do
    E=$OUT/$name/eval; [ "$T" != 512 ] && E=$OUT/$name/eval$T
    echo "=== $(date -u +%H:%M:%S) [w$W] eval $name @$T"
    $PY -m beat.eval_bridge --model Qwen/Qwen2.5-3B --adapter $OUT/$name/final_adapter --max-lora-rank 32 --max-tokens $T \
        --n 4 --out $E > $OUT/logs/eval${T}_$name.log 2>&1 || { echo "FAIL eval $name @$T"; tail -20 $OUT/logs/eval${T}_$name.log; }
    grep RESULT $OUT/logs/eval${T}_$name.log
  done
done
echo "=== $(date -u +%H:%M:%S) [w$W] XONG $JOBS"
