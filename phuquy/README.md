# Capstone — GStar Bootcamp 2026

**Cảm hứng:** BRIDGE — *Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning*
(Chen, Han, Shen, Bai, Wong · arXiv [2509.06948](https://arxiv.org/abs/2509.06948) · ICML 2026).
PDF: `related-work/01_chen2025_bridge-cooperative-sft-rl.pdf`. Paper hứa code tại
<https://github.com/ChanLiang/BRIDGE>. Kiểm 2026-11-12: repo chỉ có README, chưa có code hay dữ liệu.

**Câu hỏi nghiên cứu:** _chưa chốt_

**Trạng thái:** dựng repo, đọc paper, tìm hướng (2026-11-11)

**Môi trường** (tạo 2026-11-12, chỉ để soát dữ liệu và chạy bộ chấm trên Mac; training chạy trên cụm hoặc Kaggle):

```bash
# env chính: Math-Verify, bộ chấm của LUFFY, reward.py của nhóm
conda create -n capstone --override-channels -c conda-forge python=3.11
uv pip install --python ~/miniconda3/envs/capstone/bin/python "math-verify==0.8.0" "sympy==1.14.0" \
  "latex2sympy2_extended==1.10.2" "antlr4-python3-runtime==4.13.2" pylatexenc pandas pyarrow
# env riêng cho bộ chấm SimpleRL: latex2sympy2 ghim antlr 4.7.2, xung đột với math-verify
conda create -n capstone-simplerl-eval --override-channels -c conda-forge python=3.11
uv pip install --python ~/miniconda3/envs/capstone-simplerl-eval/bin/python latex2sympy2 sympy regex word2number numpy
```

## Tóm tắt paper (để khỏi đọc lại)

- **Vấn đề:** SFT rồi RL (hai giai đoạn) tốn kém; trộn hai loss vào một giai đoạn bằng trọng số/lịch thì có thể
  phản tác dụng, vì cập nhật SFT không phải lúc nào cũng giúp RL tăng reward.
- **Cách làm:** bài toán hai tầng. Tầng trên là "thầy" SFT, tham số là một LoRA nhỏ. Tầng dưới là "trò" RL, tham số
  là cả mô hình. Mỗi bước: (1) cập nhật mô hình bằng tổ hợp lồi của gradient SFT và gradient RL; (2) cập nhật LoRA
  để tối đa "lợi ích hợp tác" = reward khi học SFT+RL chung trừ reward khi chỉ học RL (xấp xỉ bằng một bước gradient
  RL, dùng chung rollout, không cần đạo hàm bậc hai). LoRA gộp vào mô hình khi suy luận, không tốn thêm chi phí.
- **Thiết lập:** Qwen2.5-3B, Llama-3.2-3B-Instruct, Qwen3-8B-Base. RL trên MATH phần khó (8.5K bài, mức 3–5), dữ liệu
  SFT chưng cất từ DeepSeek-R1. GRPO trong VERL, 8 rollout/prompt. Đánh giá MATH500, Minerva, OlympiadBench, AIME24,
  AMC23.
- **Kết quả:** hơn baseline mạnh nhất (SFT→RL, trộn thẳng, các cách lai một giai đoạn) khoảng 3–4 điểm trung bình,
  train ổn định hơn và nhanh hơn SFT→RL.

## Thiết lập benchmark ban đầu (2026-11-12, đã thay bằng `PLAN_baselines_qwen25_3b.md`): setup LUFFY

Chọn setup LUFFY vì nhiều paper nhất dùng chung (LUFFY, ReLIFT, SRFT, Prefix-RFT, HPT), và repo của SFT-then-RL
(arXiv 2604.23747, NeurIPS 2026) có code chạy được cho các mốc đã sửa lỗi. Setup SimpleRL của BRIDGE chỉ để đối chiếu,
vì BRIDGE không công bố dữ liệu SFT.

- **Train:** `code/data/sft_then_rl/openr1_46k_8192.parquet` (45.792 bài, mỗi bài một lời giải DeepSeek-R1 ≤ 8.192 token).
- **Đánh giá:** `code/data/sft_then_rl/all.parquet`. Toán: AIME24, AIME25, AMC (83 câu), MATH-500, Minerva, OlympiadBench.
  Ngoài phân phối: ARC-c, GPQA Diamond, MMLU-Pro. Nhiệt độ 0,6, tối đa 8.192 token, avg@32 cho AIME và AMC, chấm bằng
  Math-Verify.
- **Code tham chiếu** (chỉ đọc, gitignore): `third_party/sft_then_rl` (`e0f3573`; SFT, SFT→RL, LUFFY, ReLIFT, SRFT trên
  verl) và `third_party/LUFFY` (`01f4ae0`; script đánh giá `eval_scripts/generate_vllm.py`).
- **Trùng train–đánh giá** (`code/scripts/check_contamination.py`): AIME24, AIME25, AMC, Minerva và 3 bộ ngoài phân phối
  không trùng. MATH-500 có 4 câu, OlympiadBench 11 câu gần như y hệt một bài train (danh sách ở
  `code/data/sft_then_rl/contaminated.tsv`). Báo số trên bộ chuẩn để so được với các paper, kèm bản bỏ 15 câu này.
- **Leo bench trên `dev.parquet`** (`code/scripts/make_dev_split.py`): MATH-500 + Minerva + OlympiadBench bỏ 15 câu trùng,
  còn 1.432 câu. AIME24, AIME25, AMC và 3 bộ ngoài phân phối **chỉ chạy ở lần đánh giá cuối** của mỗi ứng viên, không
  dùng để chọn cấu hình. Lý do: AIME có 30 câu (1 câu = 3,3 điểm), nhiễu lớn hơn mọi cải tiến cần phát hiện; chọn cấu
  hình trên chính bộ báo cáo thì số báo cáo bị thổi phồng.
- **Mốc có số công bố cho model nhỏ:** LUFFY có bản Qwen2.5-Math-1.5B (TB 38,0 trên 6 bộ toán; checkpoint
  `Elliott/LUFFY-Qwen-Math-1.5B-Zero`).
- **Chưa chốt:** cỡ model và ngân sách card. Các paper dùng Qwen2.5-Math-7B trên nhiều node; cụm khoa có A100 40 GB và
  trần 2 job, nên nhiều khả năng phải chạy model nhỏ và tự chạy lại mọi mốc.

## Ý tưởng

Quy trình vòng ý tưởng (bắt chước Peer_reconstruction, 2026-11-21): skill `cs-brainstorm` → `cs-novelty-gate` → `cs-round` →
`cs-leaderboard` (`.claude/skills/`). Trạng thái, bảng baseline của nhóm, hạ tầng VM GCP: **`research/STATE.md`** — đọc trước tiên.

## Nhật ký

- 2026-11-11: tạo thư mục từ `_template/`, `git init`, tải paper BRIDGE.
- 2026-11-12: tải dữ liệu của hai bộ benchmark về `code/data/` (gitignore): `bridge/` (setup SimpleRL mà BRIDGE dùng) và
  `sft_then_rl/` (setup LUFFY, dùng chung bởi LUFFY, ReLIFT, SRFT, Prefix-RFT, HPT và arXiv 2604.23747). Nguồn và
  thiết lập đánh giá ghi trong README của từng thư mục.
- 2026-11-12: đẩy `code/data/` lên Kaggle, **private**: `nguyenlamphuquy/gstar-capstone-sft-rl-math` (29 file, giữ cấu trúc
  thư mục). Thư mục đóng gói `.work/kaggle_dataset/` (hard link); cập nhật bằng
  `kaggle datasets version -p .work/kaggle_dataset --dir-mode zip -m "<ghi chú>"`.
- 2026-11-12: tải `BaoNhat/bridge-math-curated-teachers` (bạn trong nhóm tạo, HF commit `5756b56`) về
  `code/data/bridge_curated_teachers/`: 8.519 câu MATH mức 3–5 (nhãn đã soát lại) kèm một lời giải GPT-6 Luna cho mỗi
  câu, và `reward.py` dùng chung cho mọi phương pháp. Lời giải **ngắn** (trung vị 293 ký tự), không phải chuỗi suy luận
  dài như R1.
- 2026-11-12: thử bộ chấm trên 778 lời giải GPT-6 Luna có `\boxed{<phương trình>}` (`code/scripts/boxed_equation_*.py`).
  Tỉ lệ chấm đúng: reward.py của nhóm 100%, Math-Verify của LUFFY 96,8%, bộ chấm OAT của LUFFY 96,7%,
  **bộ chấm SimpleRL 58,2%** (sai khi vế trái dài hơn 2 ký tự, như `p+q=118`). Bản thay ô bằng đáp án chuẩn: 100% ở
  cả bốn. Model học thói quen này sẽ bị chấm thấp oan trên setup SimpleRL/BRIDGE, và reward lúc RL không phạt nó.
- 2026-11-12: sửa định dạng phần SFT của bộ curated (`code/scripts/fix_curated_sft.py` →
  `code/data/bridge_curated_teachers_fixed/`, file gốc giữ nguyên): 755 ô phương trình `\boxed{p+q=118}` → `p+q=\boxed{118}`,
  320 dòng dấu công thức sai hoặc cụt; 23 dòng có đáp án là phương trình giữ nguyên. Chấm lại 8.519 dòng: reward.py
  100% → 100%, Math-Verify 99,0% → 99,2%, OAT 99,3% → 99,5%, bộ chấm SimpleRL 94,1% → 97,6%; không dòng nào tệ đi.
- 2026-11-12: **chốt cấu hình baseline mới** (`PLAN_baselines_qwen25_3b.md`): Qwen2.5-3B, hai bộ train trong
  `code/data/tracks/` (`code/scripts/make_tracks.py`): `luffy8k` (8.000 câu LUFFY, lời giải R1 ≤ 4.096 token, qua
  reward.py) và `curated` (8.519 câu của nhóm, bản sửa định dạng). Reward chung khi train: reward.py. Dev còn 1.408 câu
  (bỏ câu gần trùng với cả hai bộ). Plan cũ cho Qwen2.5-Math-1.5B bỏ vì mỗi lần RL ước 15–30 giờ.
- 2026-11-12: cập nhật Kaggle dataset lên phiên bản có `tracks/`; lệnh `datasets version` làm dataset **chuyển sang
  public** và Kaggle không cho đổi lại private. Người dùng quyết định giữ public.
- 2026-11-12: plan đổi framework train từ verl sang **Unsloth + TRL** (full fine-tuning). B1–B3 có sẵn trong TRL; LUFFY,
  SRFT, ReLIFT phải tự viết lại trên TRL và đối chiếu với bản verl của repo SFT-then-RL.
- 2026-11-12: plan đổi sang **LoRA cho mọi baseline** (r=64, alpha=64, mọi lớp tuyến tính, model gốc bf16; lr ×10 so với
  bản full fine-tune: SFT 5e-4, GRPO 1e-5). Số không so trực tiếp được với các paper full fine-tune.
- 2026-11-12: viết `code/train/` (Unsloth + TRL, LoRA: SFT, GRPO, SFT→RL, đẩy checkpoint lên HF để chạy tiếp),
  `code/evaluation/run.py` (chấm như LUFFY, prompt giống lúc train), `code/tests/` và notebook molab
  `notebooks/00_env_smoke.py` (12 bước). Trên Mac chỉ chạy test không cần model (10 qua); mọi thứ có model chạy trên molab.
  Chạy: `PYTHONPATH=code python -m train.<sft|grpo> ...` (thư mục `code` trùng tên module chuẩn của Python).
- 2026-11-12: notebook 00 trên molab (RTX PRO 6000): **mọi bước PASS**. Sửa trên đường đi: ghim vllm 0.15.1 và torchao <0.18,
  tắt TensorFlow trong transformers, tắt FlashInfer (`UNSLOTH_VLLM_NO_FLASHINFER=1`), chat template ChatML đặt tường minh,
  token kết thúc `<|endoftext|>` cho cả hai bộ, token dừng qua `vllm_sampling_params`, chia mẩu tính xác suất thủ công.
  Đo: SFT bộ A 1,8 giờ; một lượt sinh GRPO 317 s. Chốt RL 1 epoch. Viết `train/queue.py` + notebook 01 chạy 14 việc.
- 2026-11-21: mục tiêu mới: vượt bảng baseline của nhóm (curated, 512 token; mạnh nhất là GRPO-only). Dựng VM GCP A100, `code/beat/`
  (đánh giá 5 bộ BRIDGE, sinh dữ liệu SFT tự viết có/không gợi ý), quy trình vòng ý tưởng `research/` + skill `cs-*`.
