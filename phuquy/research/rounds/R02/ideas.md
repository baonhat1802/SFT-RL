# Vòng R02 — ý tưởng (cs-brainstorm), 2026-11-26
**Trọng tâm (người dùng, 26/11):** "cần novelty hơn chứ tự nhiên mỗi GRPO mix với chưng cất" — không đề xuất GRPO + λ·chưng cất.
**Thiết lập:** Qwen2.5-3B base, `curated`, `reward.py`, LoRA r32 α64, trần **2.048 token** (từ R01 bản 2), 1 seed explore.
**Code:** `loop.py` đã dùng `fastlp.py` (+25% tốc độ/rollout, đúng số — `bench_fastlp.py`).

## Bước 1 — vấn đề (số đo từ `runs/diag_cos`, `runs/diag_tokw` trên VM phuquy-a100, 26/11; model gốc, 3+2 batch × 256 câu)
| # | Vấn đề | Số đo | Bằng chứng |
|---|---|---|---|
| P1 | ~30% nhóm cả-8-sai: GRPO không có gradient trên đúng những câu cần học | allfail 0,26–0,34 | log R01, diag |
| P2 | Bắt chước thầy-có-y* trên nhóm cả-8-sai gần như vô dụng | cos(g_SD, g_RL câu khác) +0,06 vs mốc GRPO +0,15 | diag_cos |
| P3 | Giá trị của thầy phụ thuộc mạnh vào ngữ cảnh/loại mẫu, không biết trước | +0,01 (chỉ đáp án) … +0,16 (prompt đầy đủ), +0,17 (sai trong nhóm lẫn) | diag_cos |
| P4 | Chênh lệch bắt chước: văn bản viết khi biết đáp án khó học hơn văn bản tự viết | SFT gợi ý 17,7 < tự giải 19,6 dù phủ 54% > 46% | phase 1 |
| P5 | Gradient GRPO một rollout rất nhiễu | cos(g_RL A, g_RL B) 0,05–0,29 | diag |
| P6 | Trọng số từng token của thầy (RLSD) không thêm tín hiệu | tokw [0,5;2] +0,080 vs [1;1] +0,101; đảo dấu +0,084 | diag_tokw |

**Phát hiện phụ (sửa nhận định R01):** cos(g_Q, g_S) = −0,27 trong log R01 là ảo do đo tại θ′ (sau bước trong): bước theo chính gradient
GRPO cũng tụt 0,25 khi đo tại θ′ (3/3 batch). Mọi chỉ số cos trong log R01 không dùng để chẩn đoán được.

## Bước 3 — lọc: các cơ chế đã LOẠI (trùng) → `prior_art.md`
Thầy tương phản đúng/sai (RLCSD 2606.11709, 2609.21561, 2605.11609, SSOPD 2605.17497); thầy có bộ nhớ lời giải đúng cũ (Procedural Memory
Distillation 2607.01480, RLRT, ExGRPO); đồng thuận nhiều thầy (AVSD 2605.20643, CANON 2607.13643, GATES 2602.20574, 2610.40574); câu hỏi con
từ y* (SCRL 2605.22074, R³); RLSD (2604.03128, và đo: không giúp). Chỉ **2** cơ chế qua lọc — nộp đúng 2.

---

## R02-1: Rẽ nhánh tại chỗ thầy-có-y* phản đối — method `r02_fork`
Vấn đề: P1 + P2 (+P4). Câu cả-8-sai không có gradient; bắt chước thầy ở đó vô dụng (+0,06) và có chênh lệch bắt chước.
Cơ chế (lời thường): Với lời giải sai của một câu cả-nhóm-sai, cho model-có-y* chấm từng token của lời giải đó; token đầu tiên mà thầy
phản đối mạnh (log π_T − log π_S < −τ, sau một đoạn đệm) là "chỗ rẽ sai". Từ tiền tố tới ngay trước chỗ đó, chính trò (KHÔNG có y*) sinh
lại k nhánh. Nhánh nào ra đáp án đúng → nhóm nhánh có phần thưởng lẫn lộn → advantage khác 0, dồn đúng vào đoạn sau chỗ rẽ. Ví dụ: bài
tổ hợp mà trò đếm trùng ở bước 3; thầy biết lời giải mẫu chia trường hợp nên phản đối token bắt đầu cách đếm sai; rẽ từ đó, trò có cơ hội
tự chia trường hợp.
Dùng y* và phần thưởng thế nào trong cùng vòng học: y* CHỈ chọn vị trí rẽ (không vào gradient, không vào ngữ cảnh của trò); phần thưởng
của các nhánh quyết định toàn bộ chiều cập nhật.
Vì sao không quy về GRPO-only hay SFT→GRPO: tắt phần rẽ → câu cả-sai lại không có gradient; không có bước SFT/bắt chước nào.
Đối chứng novel=0: không rẽ nhánh = GRPO-only. Cờ riêng: `locator=teacher|random|entropy` (cùng số nhánh, khác cách chọn chỗ rẽ) — so
`teacher` với `random`/`entropy` là phần chứng minh vai trò của y*.
Cài đặt: 1 adapter; mỗi rollout thêm 1 forward thầy trên lời giải sai của câu cả-sai (~30% câu) + sinh k=4 nhánh × ≤2 chỗ rẽ cho ≤32 câu;
ước 1,3–1,4× GRPO-only (nhánh ngắn hơn lời giải đầy đủ). Opts: `k=4, tau=2.0, max_forks=2, max_q=32, locator=teacher`. Log: tỉ lệ đúng của
nhánh, tỉ lệ câu cả-sai "được cứu" (≥1 nhánh đúng), vị trí rẽ (% độ dài), allfail theo thời gian.
**Kiểm trước khi train (`code/beat/diag_fork.py`, chạy 26/11 trên VM):** tại model gốc, ~260 câu cả-sai, k = 8 nhánh/chỗ rẽ, so các bộ
định vị cùng ngân sách: `start` (sinh lại từ đầu — chạy thử cho thấy "cả-8-sai" vẫn được cứu ~27% khi lấy thêm mẫu, nên ĐÂY là mốc chính),
`random`, `entropy`, `depth_matched` (vị trí tương đối mà thầy chọn ở câu KHÁC — tách hiệu ứng độ sâu kiểu DEEP-GRPO), `teacher_first/min`
(thầy ngắn gọn), `teacher_full_first/min` (thầy prompt đầy đủ). Tiêu chí sống: một bộ `teacher_*` cứu nhiều câu hơn `start`, `random` và
`depth_matched` ngoài khoảng tin cậy 90%. Không đạt → bỏ ý tưởng.
Dự đoán (viết trước khi chạy): kiểm trước — tỉ lệ cứu `teacher` ≥ 1,5× `random`; train — allfail giảm nhanh hơn GRPO-only, Δ TB dev +1…+2,5.
```
Lọc cs-novelty-gate — r02_fork — QUA
G1 P1 allfail 0,26–0,34 + P2 cos +0,06 · đường nhân quả: rẽ đúng chỗ sai → nhánh có phần thưởng lẫn lộn → gradient trên câu trước đây 0
G2 gần nhất BREAD 2506.17211 (rẽ từ tiền tố LỜI GIẢI CHUYÊN GIA, chọn theo tỉ lệ đúng; ở đây rẽ trên quỹ đạo CỦA TRÒ, chọn theo bất đồng
   của thầy-có-y*), DEEP-GRPO 2602.14169 (rẽ trên quỹ đạo sai, utility không dùng y*), TreeAdv/LATR/TEMPO (theo entropy), T-STAR 2604.07165
   (tương phản nhánh đúng/sai tại chỗ phân kỳ, không định vị bằng y*) · truy vấn: "branch rollouts ... reference solution disagreement",
   "GRPO tree rollouts branching critical tokens", "DEEP-GRPO pivots" · ĐÃ ĐỌC phương pháp (26/11): DEEP-GRPO chọn pivot theo
   Q(t) ∝ P_φ(thành công | t/T)·(t/T)^γ, P_φ = hồi quy logistic CHỈ trên độ sâu tương đối, học từ kết quả nhánh — không dùng y*, không
   entropy; 1 pivot/quỹ đạo sai, 8 nhánh, loss che tiền tố. BREAD rẽ từ tiền tố LỜI GIẢI CHUYÊN GIA (ranh giới bước), độ dài gợi ý tìm
   nhị phân theo p_correct; tiền tố chuyên gia không tính loss. → bộ định vị bằng bất đồng thầy-có-y* trên quỹ đạo của trò là phần mới.
G3 y* chọn chỗ rẽ trong cùng rollout; phần thưởng các nhánh cho gradient
G4 chỉ Qwen2.5-3B (chính nó có y* trong ngữ cảnh) · G5 phần thưởng nhị phân giữ nguyên; y* không vào ngữ cảnh trò → không rò
G6 ~1,3–1,4× · G7 novel=0 = GRPO-only; locator=random/entropy là đối chứng cùng ngân sách · G8 neo: locator=random phải ≥ GRPO-only;
   kiểm trước tại model gốc trả lời "y* có định vị được không" trước khi train
```

## R02-2: Chọn cấu hình hướng dẫn theo độ chuyển giao đo tại chỗ — method `r02_transfer`
Vấn đề: P3 (+P5). Giá trị của thầy dao động +0,01…+0,17 theo ngữ cảnh/loại mẫu, không biết trước, và đổi theo thời gian.
Cơ chế (lời thường): mỗi rollout chia câu thành hai nửa A/B; với mỗi cấu hình hướng dẫn c (ngữ cảnh thầy × loại mẫu), ước độ chuyển giao
bậc nhất τ_c = cos(g_c(A), g_RL(B)) theo kiểu cross-fitting (không dùng chung mẫu → không bị thổi phồng); giữ EMA của τ_c, chọn cấu hình
theo bandit (UCB trên EMA), bật hướng dẫn chỉ khi cận dưới của τ_c > 0.
Dùng y* và phần thưởng: y* trong ngữ cảnh thầy; gradient phần thưởng trên nửa câu khác quyết định cấu hình nào được dùng.
Vì sao không quy về GRPO-only: mọi cấu hình bị tắt khi τ ≤ 0 → đúng là GRPO-only (neo).
Đối chứng novel=0: GRPO-only. Cờ: `select=bandit|fixed_best|uniform`.
Cài đặt: thêm ~3–4 lượt backward/rollout → ~1,6–1,8× → cần GRPO-only cùng ngân sách.
Dự đoán: Δ TB dev 0…+1,5; rủi ro lớn: nhiễu (mốc 0,05–0,29) làm bandit chọn ngẫu nhiên.
```
Lọc cs-novelty-gate — r02_transfer — QUA (yếu)
G1 P3 +0,01…+0,17 · đường nhân quả: chỉ dùng hướng dẫn có độ chuyển giao dương đo được
G2 gần: Learning to Reweight Examples (Ren 2018, gradient trên validation), auxiliary-loss gating theo cosine (Du 2018), multi-teacher OPSD
   2610.40574 (chọn thầy theo KL + bỏ phiếu, không theo chuyển giao phần thưởng) · khác: tín hiệu chọn = cross-fitted RL transfer · độ mới THẤP
G3 y* (thầy) + phần thưởng (B) cùng quyết định bước · G4 ok · G5 ok · G6 1,6–1,8× → cần đối chứng cùng ngân sách · G7 ok · G8 neo: τ≤0 → GRPO
LƯU Ý: chạm workstream cân trọng số SFT/RL của Bao Nhat & Huy Hoang — hỏi nhóm trước.
```

---
## KẾT QUẢ KIỂM TRƯỚC R02-1 (26/11, `runs/diag_fork` trên VM, 262 câu cả-8-sai, k = 8) — LOẠI
| bộ định vị | câu được cứu | KTC 90% | vị trí |
|---|---|---|---|
| start (sinh lại từ đầu) | 23,7% | 19,5–28,2 | 0 |
| depth_matched | 24,0% | 19,8–28,6 | 0,24 |
| teacher_min | 21,4% | 17,6–25,6 | 0,35 |
| teacher_first | 18,3% | 14,5–22,1 | 0,24 |
| teacher_full_first | 17,6% | 13,7–21,4 | 0,29 |
| random | 15,3% | 11,8–19,1 | 0,46 |
| teacher_full_min | 15,3% | 11,8–19,1 | 0,38 |
| entropy | 11,1% | 7,6–14,1 | 0,59 |
Không bộ `teacher_*` nào hơn `start`/`depth_matched`; ở cùng độ sâu 0,24, thầy (18,3%) còn kém vị trí mượn từ câu khác (24,0%) → bất đồng
của thầy-có-y* KHÔNG định vị chỗ rẽ sai. Rẽ càng sâu càng kém (tiền tố lời giải sai là gánh nặng với model 3B). Dự đoán "teacher ≥ 1,5×
random" SAI về ý nghĩa (1,2–1,4× random nhưng < start). Phát hiện phụ: ~24% câu "cả-8-sai" được cứu chỉ bằng 8 mẫu nữa từ đầu → phần lớn
nhóm cả-sai là do lấy mẫu chưa đủ.

---
## R02-3: Thầy "y* trong trọng số" — LOẠI ở kiểm trước (27/11, `runs/diag_wt`)
Cơ chế: với mỗi câu, θ′ₓ = θ − ε·∇NLL(y*|x)/‖·‖ (một bước SFT trên adapter), θ′ₓ chấm lời giải của trò (không có y* trong ngữ cảnh); lý do:
model base không đọc y* trong ngữ cảnh (~1% token đổi). Rà văn liệu (3 truy vấn): chưa thấy; gần: OPSD, ATESD 2605.11458, PAINT, RLT 2506.08388,
fast weights, one-shot imitation qua MAML. Kết quả (mốc GRPO +0,14): ε 0,3/1/3 → token đổi > 1 nat 0,7%/21%/89%, chuyển giao −0,16…−0,22 ở
mọi ε và cả hai loại mẫu; gradient SFT trên y* −0,18. Tiêu chí (phản ứng mạnh hơn VÀ chuyển giao ≥ +0,16) không đạt → bỏ. Lý do: thầy-trọng-số
≈ hướng SFT trên văn bản y*, mà hướng này chống phần thưởng với model base.

---
## Chẩn đoán 27/11 — vì sao SFT trên y* chống phần thưởng (`runs/diag_ystar`)
Tách gradient SFT-y* theo loại token: chữ −0,13 (NLL 2,03), ký hiệu toán −0,12, số −0,06, `\boxed` ≈ 0; SFT trên lời giải đúng của chính trò +0,07
(mốc GRPO +0,08); cos(SFT-tự-thân, SFT-y*) −0,12. Kết luận: thứ SFT học từ y* là GIỌNG VĂN của Luna, ngược hướng giọng văn của chính trò mà phần
thưởng đang củng cố. Ý "chỉ bắt chước kế hoạch (che số)" LOẠI luôn (mọi loại token ≤ 0). Mọi tín hiệu dương đều nằm trên văn bản của chính trò.
