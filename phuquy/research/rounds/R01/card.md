# Thẻ vòng R01 — dự đoán viết TRƯỚC khi chạy (2026-11-21)
Thiết lập: `code/beat/loop.py` (vòng GRPO riêng; mọi run cùng khung), Qwen2.5-3B, `curated`, LoRA r32 α64, lr 1e-5, 8 lời giải/câu,
128 câu/rollout, 2 bước tối ưu/rollout, 1 epoch (~66 rollout), trần 512 token, seed 1. Đánh giá `beat.eval_bridge` n = 4 @512:
**DEV** (MATH500 + Minerva + OlympiadBench) để xếp hạng; TB 5 bộ chỉ tham khảo. Thẻ: `ideas.md`. Luật đẩy lên: skill cs-leaderboard
(Δ DEV vs novel=0 ≥ ngưỡng, DEV ≥ GRPO-only + ngưỡng, chỉ số cơ chế đúng chiều). Ngưỡng tạm ≥ 2,0 (`research/explore.md`).

Mốc đã đo (cùng cách chấm): model gốc @512 TB 16,8 ± 1,0 · cắt 60,0% (n = 4).

| run | DEV dự đoán so với grpo_s1 | chỉ số cơ chế — chiều dự đoán |
|---|---|---|
| grpo_s1 (novel=0 chung) | — (dự đoán DEV tuyệt đối: base + 4…8) | allfail giảm từ ~0,3; cắt 60% → 20–30% |
| metaopsd_s1 | +2…+4 | cos_gQ_gS tăng dần, > 0; hess_rel 0,05–0,5 |
| metaopsd_h0_s1 | +1…+2 | — |
| metaopsd_same_s1 | thấp hơn metaopsd_s1 | — |
| guide_tokw_s1 | +1,5…+3 | hint_correct 0,2–0,5; allfail giảm nhanh hơn grpo |
| guide_s1 (Guide, mốc) | +1…+2 | — |
| opsd_s1 | +0…+2 | kl_ST giảm |
| opsd_dense_s1 | 0…+2 | kl_ST giảm; cắt giảm nhanh hơn grpo |
| potential_s1 | +1…+2,5 | Vlast_mean tăng nhanh hơn V0_mean |
| fade_s1 | +1…+2,5 | fade_level_mean tăng rồi về gần 0 |
| compress_sd_s1 | +1…+2 (@512) | cắt giảm nhanh hơn grpo |
| grpo_2ep_s1 (cùng ngân sách metaopsd) | +0…+1,5 | — |
Điểm neo: GRPO bản Unsloth (`code/beat/grpo_only.sh`, cấu hình bảng nhóm) — grpo_s1 phải cách nó ≤ ngưỡng; lệch hơn → khung có lỗi.
Rủi ro đã biết: alpha của bước trong (metaopsd) chưa chỉnh — bước quá nhỏ thì metaopsd ≈ grpo; quá lớn thì bước trong phá model.
Ghi ‖Δθ‖ nếu cần chỉnh, ghi vào nhật ký dưới đây.

## Nhật ký chạy
- 2026-11-21 04:27 UTC: chạy trên cụm BKPT, seat 85148 (2× A100-PCIE gpu02). Lượt 1 hỏng do gpu02 thiếu `libcuda.so` (Triton) → symlink
  trong /tmp + `TRITON_LIBCUDA_PATH`. Smoke 7/7 không lỗi. Rollout đầu của grpo_s1 (128×8): allfail 0,59, cắt 0,54, 162 s/rollout.
- Smoke metaopsd với `alpha = 1e-3` cố định: hess_rel ~1e-3, bước trong gần như bằng 0 → metaopsd ≈ GRPO. Sửa TRƯỚC khi chạy thật:
  bước trong chuẩn hoá `inner_norm = 0,1` (≈ 1,3 bước Adam của LoRA ~60M tham số). Dự đoán giữ nguyên. Thứ tự: metaopsd chạy ngay sau
  grpo_s1 trên card 1 (tmux `capmeta`), hàng đợi chính tiếp tục trên card 0 sau guide_tokw_s1.
- Quan sát smoke: cos(g_Q, g_S) ≈ 0 (−0,002; 0,08) — hướng lời dạy của thầy gần trực giao với hướng GRPO.
- 2026-11-21 ~05:30 UTC: prompt chấm của thầy đổi sang bản ngắn gọn (`TEACHER_TEMPLATE`: đề + lời giải mẫu + "giải gọn, giữ bước quan
  trọng") cho opsd, opsd_dense, metaopsd, compress_sd và phần tokw; bỏ chưng cất trên lời giải bị cắt (`sd_skip_trunc=1`). Lý do: prompt cũ
  (HINT_TEMPLATE, "viết kỹ, bổ sung bước") làm thầy kéo trò viết dài — ngược áp lực 512 token, và ngược hẳn mục đích của compress_sd.
  Prompt SINH lời giải có gợi ý (guide) giữ HINT_TEMPLATE. **guide_tokw_s1 đang chạy dùng prompt chấm cũ** (bắt đầu 04:49, trước khi sửa).
  `--opt teacher=full` khôi phục prompt cũ nếu cần ablation. Dự đoán giữ nguyên.
- **2026-11-21 05:21 UTC — R01 bản 2 (người dùng chốt: bỏ trần 512 khi train).** Mọi run train với trần **2.048 token**
  (`--max-completion 2048 --micro-batch 4 --vllm-mem 0.4`), đánh giá @512 **và** @2048. Bản 512 dừng giữa chừng (grpo_s1 ~20/66
  rollout, guide_tokw_s1 ~6/66) → `runs/r01_512/` trên gpu02, log `cap_r01_512.log`, `experiments/rounds/R01/*.log.jsonl` (phần đã kéo).
  Thêm mốc: model gốc @2048 (bảng nhóm: 22,9). Ưu tiên: grpo_s1 (card 0), metaopsd_s1 (card 1, sau smoke @2048), rồi hàng đợi chính theo
  jobs.txt. Ước 6–8 giờ/run → các ablation cuối hàng có thể không kịp trong seat (hết 2026-11-23 04:09 UTC).
  Dự đoán Δ giữ nguyên; dự đoán tuyệt đối về % bị cắt không còn áp dụng ở trần 2.048.
- 2026-11-21 ~06:30 UTC: 4 ablation (guide_s1, metaopsd_h0_s1, metaopsd_same_s1, grpo_2ep_s1) chuyển sang **molab** (người dùng) —
  không đủ giờ trong seat 85148 ở 2.048 token. Chạy bằng `notebooks/41_r01_molab.py` → `beat.molab_run` (checkpoint lên HF repo
  `<HF_USER>/capstone-r01-<job>` mỗi 5 rollout, chạy tiếp qua phiên). Trên cụm các job này bị giữ bằng `.claim` (where=molab).
  Lưu ý khi so: molab là RTX PRO 6000 (Blackwell, FLASH_ATTN), cụm là A100 — cùng code, khác phần cứng; micro-batch 8 (cụm 4).
- **2026-11-21 07:04 UTC — metaopsd_s1 SỤP, dừng ở rollout 8/66** (kết quả âm tính, log `experiments/rounds/R01/metaopsd_s1.2048.log.jsonl`).
  Phần thưởng 0,27 → 0,25 (r3) → 0,11 (r5) → 0,02 (r6–7); câu cả-nhóm-sai 35% → 89%; ‖∇L_SD‖ 0,06 → 2,6. GRPO cùng lô câu: 0,38 ở r4–7.
  cos(g_Q, g_S) âm suốt (−0,15…−0,36): mỗi bước học từ thầy-có-y* làm GIẢM phần thưởng trên câu khác. Thiết kế giữ bước trong
  (`keep_inner=1`, 2 bước chưng cất ≈1,3 bước Adam mỗi rollout) đưa phần có hại vào model nhanh hơn số hạng meta (chặn ở 1×) bù được.
  → Dự đoán "+2…+4" SAI; cos dự đoán tăng dần và > 0, thực tế âm suốt.
  Chạy tiếp **metaopsd_v3_s1** = MAML chuẩn (`keep_inner=0`: bước trong chỉ để nhìn trước, model chỉ đi theo meta-gradient).
  Dự đoán v3 (viết trước khi có số): Δ DEV vs grpo_s1 **−1…+2**; không sụp; cos vẫn âm lúc đầu.
- 07:40 UTC: ablation molab đổi sang MAML chuẩn và đổi tên: `metaopsd_v3_h0_s1` (keep_inner=0, hessian=0), `metaopsd_v3_same_s1`
  (keep_inner=0, query=same) — so với metaopsd_v3_s1. Tên mới để không dùng nhầm checkpoint HF của bản cũ.
- **2026-11-21 ~12:45 UTC — grpo_s1 xong (mốc của vòng).** Train 66 rollout @2.048, phần thưởng train 0,27 → ~0,60.
  @2.048: TB **32,4±0,7**, DEV **39,9±0,4**, cắt 4,0% (model gốc @2.048: 26,4 / 33,9 / 5,2%) → **+6,0 TB, +6,0 DEV**.
  @512: TB 17,6, DEV 23,8, cắt 68,8% (gốc 16,8 / 23,6 / 60%) — train dài làm trò viết dài, nên @512 không dùng để so các run 2.048.
  Tóm tắt: `experiments/rounds/R01/grpo_s1/`. Card 0 chuyển sang guide_tokw_s1 (prompt chấm ngắn gọn, bản 2.048).
  metaopsd_v3_s1 r25/66: không sụp, nhưng phần thưởng train ~0,36 so với grpo ~0,53 cùng rollout; cos(g_Q,g_S) vẫn ≈ −0,3.
- **2026-11-21 14:05 UTC — metaopsd_v3_s1 TẠM DỪNG ở rollout 33/66 (người dùng chốt), sụp chậm.** Phần thưởng train 0,38 (r25) →
  0,25 (r32), dưới cả lúc đầu (0,27); GRPO cùng rollout ~0,53. cos(g_Q,g_S) −0,35 → −0,50, ngày càng ngược. Cùng cơ chế với
  metaopsd_s1 (keep_inner=1), chỉ chậm hơn → MAML chuẩn không cứu được hướng thầy-có-y* ngược reward. Dự đoán v3 "không sụp" SAI.
  Checkpoint lưu ở `/media02/lhbac22/projects/capstone/ckpt_r01/metaopsd_v3_s1/`: `ckpt/` = trạng thái đủ để chạy tiếp (sau rollout
  30: adapter + optim.pt + levels.npy), `adapter_latest/` = trọng số sau r32 (đánh giá được, không chạy tiếp được). Chạy tiếp: chép
  `ckpt/` về `runs/r01/metaopsd_v3_s1/ckpt/` rồi chạy lại cùng lệnh beat.loop (`--opt keep_inner=0`), loop tự nối từ ckpt.
  Log: `experiments/rounds/R01/metaopsd_v3_s1/`. Card 1 chuyển sang opsd_s1 (14:07 UTC).
