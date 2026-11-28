# Thẻ vòng R03 (2026-11-27)
Run: `cfpair_s1` (`--idea cfpair`, lam=0,1, pert=numbers), seed 1, 1 epoch curated (66 rollout × 128 câu × 8 mẫu), train @2.048, LoRA r32 α64,
lr 1e-5 — cùng cấu hình và thứ tự câu với `grpo_s1` (R01). Khác duy nhất: số hạng tương phản. Đánh giá @512 và @2.048, n = 4.
Code mới dùng đường `fastlp` (+ vLLM sleep) — đã kiểm khớp số với đường cũ của grpo_s1 (`bench_fastlp.py`, ≤ 2e-6).
| run | dự đoán (viết TRƯỚC khi chạy) |
|---|---|
| cfpair_s1 | TB 5 bộ @2048: Δ so với grpo_s1 **0…+1**; phần thưởng train ≥ grpo_s1 từ nửa sau; `cf_margin` tăng dần (bắt đầu ~0,19 nat/token); độ dài và tỉ lệ cắt ≈ grpo_s1 (không kéo giọng văn Luna) |
Rủi ro: số hạng −NLL(ỹ*) không chặn; mỗi câu chỉ xuất hiện 1 lần/epoch nên chưa chặn. Nếu `cf_margin` tăng vọt (> 2) hoặc độ dài lệch mạnh → dừng.

## Kết quả (27/11): SỤP, dừng ở rollout 30/66
Dự đoán "Δ 0…+1, cf_margin tăng dần, độ dài ≈ grpo" — SAI ở nửa sau: rollout 0–19 ngang/nhỉnh (+0,006), từ 20 tụt, 25–29 phần thưởng 0,403 vs 0,543;
cf_margin 0,19 → 1,08 không chặn; độ dài giảm ~150 token nhưng cắt vẫn cao hơn. Rủi ro đã ghi ("−NLL(ỹ*) không chặn") là nguyên nhân khả dĩ nhất.
Không đánh giá (run không xong). Bài học: kiểm trước bậc nhất tại θ0 không bắt được hỏng hóc tích luỹ — cần dạng có chặn trước khi train.

## R03-1 bản có chặn — `cfpair_reg_s1` (chạy 27/11)
`--opt form=dpo --opt lam=0.2 --opt beta=10`: h = [TB logp(y*) − ref] − [TB logp(ỹ*) − ref], ref = model gốc (adapter tắt); loss λ·TB softplus(−βh)/β.
Tại h = 0 gradient = đúng bản tuyến tính λ = 0,1 đã kiểm trước; trọng số σ(−βh) còn 0,27 ở h = 0,1 và 0,05 ở h = 0,3 (bản cũ sụp khi margin tăng thêm
~0,6–0,9). Chốt chặn tự động: TB phần thưởng 5 rollout gần nhất < grpo_s1 − 0,05 → dừng.
Dự đoán (viết trước): không sụp; cf_h dừng ở ~0,2–0,3, cf_w về ~0,05–0,1 sau ~10 rollout (số hạng tự tắt dần); phần thưởng ≈ grpo_s1 ± 0,02;
Δ TB đánh giá 0…+1 (nhiều khả năng ≈ 0 vì tín hiệu tắt sớm).
