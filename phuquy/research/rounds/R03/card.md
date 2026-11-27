# Thẻ vòng R03 (2026-11-27)
Run: `cfpair_s1` (`--idea cfpair`, lam=0,1, pert=numbers), seed 1, 1 epoch curated (66 rollout × 128 câu × 8 mẫu), train @2.048, LoRA r32 α64,
lr 1e-5 — cùng cấu hình và thứ tự câu với `grpo_s1` (R01). Khác duy nhất: số hạng tương phản. Đánh giá @512 và @2.048, n = 4.
Code mới dùng đường `fastlp` (+ vLLM sleep) — đã kiểm khớp số với đường cũ của grpo_s1 (`bench_fastlp.py`, ≤ 2e-6).
| run | dự đoán (viết TRƯỚC khi chạy) |
|---|---|
| cfpair_s1 | TB 5 bộ @2048: Δ so với grpo_s1 **0…+1**; phần thưởng train ≥ grpo_s1 từ nửa sau; `cf_margin` tăng dần (bắt đầu ~0,19 nat/token); độ dài và tỉ lệ cắt ≈ grpo_s1 (không kéo giọng văn Luna) |
Rủi ro: số hạng −NLL(ỹ*) không chặn; mỗi câu chỉ xuất hiện 1 lần/epoch nên chưa chặn. Nếu `cf_margin` tăng vọt (> 2) hoặc độ dài lệch mạnh → dừng.
