# Vòng R01 — ý tưởng (cs-brainstorm)
Ngày 2026-11-21. **Trọng tâm (người dùng chốt):** kết hợp SFT + GRPO kiểu meta-learner/hai tầng như BRIDGE nhưng thầy là **tự chưng cất
on-policy** (chính model, có lời giải Luna `y*` trong ngữ cảnh); ghép nhiều bài được (G2 nới 2026-11-21), mỗi thành phần có cờ tắt riêng.
Thiết lập: Qwen2.5-3B base, `curated` 8.519 câu, `reward.py`, LoRA r32 α64, trần 512 token, explore 1 seed (`research/explore.md`).

## Bước 1 — vấn đề (có số đo + bằng chứng)
| # | Vấn đề | Số đo | Bằng chứng |
|---|---|---|---|
| P1 | Câu mà cả 8 lời giải đều sai cho advantage 0 → GRPO không bao giờ học được câu đó | tỉ lệ nhóm cả-8-sai mỗi bước | cấu trúc; smoke 64 câu: model gốc đúng 36% mẫu, 39% câu không có lời giải đúng trong 4 mẫu (8 mẫu: ước ~30%) |
| P2 | Học từ văn bản Luna lệch phân phối → model tụt | TB sau SFT | bảng nhóm 14,4 → 8,1; SFT→GRPO 17 < GRPO 22 |
| P3 | Trần 512 token: phần lớn lợi ích GRPO là viết gọn, chưa phải giải giỏi hơn | % bị cắt; TB @512 vs @2048 | base cắt 59–67%; GRPO@512 22,3 ≈ base@2048 22,9 (bảng nhóm) |
| P4 | Tín hiệu thưa: một phần thưởng cho cả lời giải, mọi token cùng advantage | — (cấu trúc) | GRPO; RLSD/CAST/HSD đều nhắm vào đây |
| P5 | Thầy có `y*` có thể dạy trò "nhảy tới đáp án" (rò) và ép đổi mọi token | KL thầy–trò, độ dài, TB | RLSD (2604.03128) báo OPSD thuần rò + không ổn định; 2606.11173: thầy có lời giải mẫu ép đổi hành vi ở mọi token |
| P6 | Thầy tốt hơn không có nghĩa là trò học tốt hơn — không ai đo lời dạy có giúp trò không | Δ phần thưởng của trò sau một bước | 2610.10447 (JOLT) nêu cùng nhận xét; BRIDGE đo bằng "lợi ích hợp tác" |

## Bước 2–3 — cơ chế + lọc (G2 nới: ghép được, mỗi phần có cờ tắt)
| Cơ chế | Vấn đề | Kết quả |
|---|---|---|
| Meta-learning kiểu MAML: θ được chọn để một bước học từ thầy-có-y* làm tăng phần thưởng trên câu KHÁC (bậc hai qua sai phân) | P6, P2, P1 | **QUA** → R01-1 (bản LoRA thầy kiểu BRIDGE bị thay theo người dùng) |
| Gợi ý `y*` vào prompt chỉ cho câu cả-nhóm-sai + hiệu chỉnh trọng số (Guide) | P1 | **mốc** (chạy lại nguyên Guide = G2a) → `jobs.txt` |
| Guide + trọng số từng token do thầy-có-`y*` quyết định, chiều do phần thưởng quyết định (RLSD) | P1 + P4 + P5 | **QUA** (ghép) → R01-2 |
| Gợi ý rút dần: mỗi câu giữ một tỉ lệ tiền tố `y*`, giảm khi trò tự giải được | P1 | **QUA** (gần Prefix-RFT, QuestA, Adaptive Guidance) → R01-3 |
| Thầy-có-Luna làm thầy *viết gọn*: chỉ chưng cất trên lời giải ĐÚNG nhưng DÀI | P3 | **QUA** → R01-4 |
| Đổi phần thưởng thêm phạt độ dài | P3 | LOẠI — G2d (chỉnh tham số) + G5 (gaming) |
| SFT trên dữ liệu "gợi ý rồi tự viết" rồi GRPO | P2 | LOẠI làm ý tưởng — hai giai đoạn (G3); là **điểm neo** phase 1 |

Truy vấn đã chạy (WebSearch 2026-11-21): "bilevel meta-learning self-distillation teacher conditioned on reference solution GRPO",
"GRPO all rollouts incorrect add hint reference solution in prompt … importance weight", "teacher updated by student improvement Meta Pseudo
Labels language model reasoning RL privileged context teacher LoRA". Đọc tóm tắt + phương pháp JOLT (2610.10447). Không thấy công thức hai
tầng nào cho tự chưng cất.

---

## R01-1: Meta-OPSD kiểu MAML — method `r01_metaopsd`
_Sửa 2026-11-21 theo người dùng ("meta learning kiểu MAML"), TRƯỚC khi chạy: bỏ LoRA thầy kiểu BRIDGE; tham số meta là chính θ._
**Vấn đề:** P6 + P2 + P1. Lời dạy của thầy (OPSD, thầy = chính model có `y*`) chỉ có ích nếu nó **chuyển sang câu chưa được dạy** — lúc thi
không có `y*`. OPSD/JOLT không đo điều đó; BRIDGE đo nhưng thầy là văn bản Luna (P2) và né bậc hai.
**Cơ chế (lời thường):** Mỗi rollout chia câu thành support S (thầy dạy, ưu tiên câu cả-nhóm-sai) và query Q (câu khác, chấm bằng phần
thưởng). Bước trong: học một bước từ thầy trên S. Bước ngoài: đo phần thưởng GRPO của model ĐÃ học bước đó trên Q, và đạo hàm ngược qua
bước trong (MAML bậc hai). Ví dụ: thầy dạy một bài tổ hợp; nếu cách dạy đó làm model giải tệ đi trên các bài khác (vd dạy "nhảy tới đáp
án"), số hạng bậc hai đẩy θ ra khỏi vùng mà lời dạy kiểu đó có tác dụng.
```
θ' = θ − α ∇L_SD(θ; S)                    L_SD = Σ_t KL( π_θ(·|x,y_<t) ‖ sg π_θ(·|x,y*,y_<t) ) trên lời giải trò tự sinh
max_θ J_RL(θ'; Q)
∇_θ J_RL(θ') = g_Q − α H_S g_Q,   g_Q = ∇_θ' J_RL(θ'; Q),   H_S g_Q ≈ [∇L_SD(θ+εg_Q) − ∇L_SD(θ−εg_Q)] / 2ε
cập nhật: θ ← θ' + β (g_Q − α H_S g_Q)    (giữ bước trong: lúc thi không có bước thích nghi)
```
**Dùng y* và phần thưởng thế nào trong cùng vòng học:** `y*` chỉ ở ngữ cảnh thầy trong bước trong (S); phần thưởng ở bước ngoài (Q), đạo
hàm ngược qua bước trong.
**Vì sao không quy về GRPO-only hay SFT→GRPO:** tắt bước trong → GRPO-only; bỏ bậc hai → FOMAML (≈ xen kẽ OPSD rồi GRPO).
**Đối chứng:** `novel=0` = GRPO-only · `hessian=0` (FOMAML) · `query=same` (Q = S: lời dạy có chuyển sang câu khác không) · `meta=0`
(OPSD + GRPO cộng thẳng, không nhìn trước).
**Cài đặt:** 1 adapter (LoRA r32 của trò). Mỗi rollout thêm: forward thầy trên S, backward bước trong, backward GRPO tại θ' (tỉ lệ
π_θ'/π_θ từ rollout của θ), 2 backward sai phân trên S. Ước **1,5–1,8× GPU-giờ** GRPO-only → kèm GRPO-only cùng ngân sách (G6). Chỉ số:
cos(g_Q, −∇L_SD) (lời dạy cùng hướng phần thưởng?), ‖α H_S g_Q‖/‖g_Q‖ (bậc hai có đáng kể?), tỉ lệ nhóm cả-8-sai, KL thầy–trò.
**Dự đoán (viết trước):** Δ TB dev vs novel=0 **+2…+4**; `hessian=0` +1…+2; `query=same` thấp hơn bản đủ (lời dạy học thuộc câu S);
cos(g_Q, −∇L_SD) tăng dần theo bước.
```
Lọc cs-novelty-gate — r01_metaopsd — QUA
G1 P6+P2+P1 · đường nhân quả: meta-gradient bậc hai qua bước học-từ-thầy → θ được chọn để lời dạy trên S làm tăng phần thưởng trên Q
G2: gần nhất BRIDGE (nhìn trước một bước nhưng phép phạt né bậc hai; thầy = văn bản Luna), AMFT (meta-gradient cho một μ chung), JOLT
    (không hai tầng), MAML/DARTS (khung meta-learning + sai phân Hessian, ngoài LLM) · ghép MAML + OPSD + GRPO: được (nới)
G3 y* (bước trong) và phần thưởng (bước ngoài) cùng một cập nhật · G4 ok · G5 chọn trên dev; rò: query là câu khác
G6 1,5–1,8× → GRPO-only cùng ngân sách · G7 novel=0, hessian=0, query=same, meta=0 · G8 novel=0 = GRPO-only; α=0 → GRPO-only trên Q
```

## R01-2: Gợi ý cho câu cả-nhóm-sai + trọng số từng token từ thầy — method `r01_guide_tokw`
**Vấn đề:** P1 (câu khó không có gradient) + P4 (tín hiệu thưa) + P5 (rò).
**Cơ chế (lời thường):** Câu mà cả 8 lời giải sai → sinh thêm 4 lời giải với `y*` trong prompt (Guide), chấm, đưa vào nhóm; log-xác suất của
chúng tính dưới prompt **không** gợi ý, trọng số quan trọng có cắt. Thêm: advantage từng token `A_{i,t} = A_i · w_t`, với `w_t` lớn ở token mà
thầy-có-`y*` và trò bất đồng nhiều (chuẩn hoá trung bình 1, cắt [0,5; 2]) — chiều cập nhật vẫn do phần thưởng (RLSD), nên thầy không kéo trò
về một lời giải sai.
**Đối chứng:** `novel=0` = GRPO-only · `hint=0` (bỏ lời giải có gợi ý), `tokw=0` (bỏ trọng số từng token; `tokw=0` = Guide).
**Cài đặt:** 1 adapter; sinh thêm ~4 lời giải cho ~30% câu + 1 forward có `y*`; ước **1,3–1,5×**. Chỉ số: tỉ lệ nhóm cả-8-sai, tỉ lệ lời
giải có gợi ý đúng, phân bố `w_t`.
**Dự đoán:** Δ vs novel=0 **+1,5…+3**; `tokw=0` (Guide) +1…+2; tỉ lệ nhóm cả-8-sai giảm.
```
Lọc — r01_guide_tokw — QUA (ghép Guide 2506.13923 + RLSD 2604.03128; gần OC-GRPO, HiLL 2604.00698, SAGE 2602.03143, GCPO 2510.07790)
G3: lời giải có gợi ý (y*) vào nhóm GRPO cùng phần thưởng · G4 ok · G5 log-prob dưới prompt không gợi ý · G6 1,3–1,5× · G7 hint=0, tokw=0 · G8 novel=0 = GRPO-only
```

## R01-3: Gợi ý rút dần theo từng câu — method `r01_fade`
**Vấn đề:** P1. Gợi ý cả `y*` thì dễ quá (rò); không gợi ý thì cả nhóm sai.
**Cơ chế (lời thường):** Mỗi câu giữ một mức gợi ý `p_x ∈ {0; 25; 50; 75%}` = phần đầu của `y*` đặt sẵn ở đầu câu trả lời. Câu cả-nhóm-sai
→ tăng `p_x` một nấc lần sau; nhóm có ≥ 2 lời giải đúng → giảm một nấc. Token của tiền tố không tính loss; phần model tự viết tiếp là
on-policy. Như thầy dạy kèm: đưa vài dòng đầu, học trò làm được thì lần sau đưa ít hơn.
**Đối chứng:** `novel=0` (p_x luôn 0) = GRPO-only · `adapt=0` (p_x cố định 50% cho câu cả-nhóm-sai).
**Cài đặt:** 1 adapter, lưu 8.519 số nguyên; chi phí ≈ 1,0–1,1×. Chỉ số: phân bố `p_x` theo thời gian, tỉ lệ nhóm cả-8-sai.
**Dự đoán:** Δ +1…+2,5; phân bố `p_x` dồn về 0 cuối train (nếu không → trò phụ thuộc gợi ý → TB không tăng).
```
Lọc — r01_fade — QUA (gần Prefix-RFT, QuestA, Adaptive Guidance 2506.13923; khác: nấc gợi ý theo từng câu, tăng/giảm theo kết quả nhóm)
G3 tiền tố y* + phần thưởng trong cùng rollout · G5 tiền tố không tính loss · G6 ~1× · G7 adapt=0 · G8 novel=0 = GRPO-only
```

## R01-4: Thầy-có-Luna dạy viết gọn — method `r01_compress_sd`
**Vấn đề:** P3. Ở trần 512, model gốc bị cắt 59–67%; lời giải Luna ngắn (trung vị 115 token) — đây là cái Luna giỏi mà SFT thẳng làm hỏng.
**Cơ chế (lời thường):** Chỉ chưng cất từ thầy-có-`y*` trên những lời giải **đúng nhưng dài** (dài hơn trung vị nhóm). Thầy đã thấy một lời
giải ngắn nên xác suất của thầy thấp ở các bước thừa → trò học bỏ bước thừa mà vẫn đúng. Không đụng lời giải sai (để GRPO lo), nên không rò đáp án.
**Đối chứng:** `novel=0` = GRPO-only · `longonly=0` (chưng cất trên mọi lời giải đúng).
**Cài đặt:** 1 adapter + 1 forward có `y*` cho ~1/3 lời giải; ≈ 1,2×. Chỉ số: % bị cắt, số token TB, TB @512 vs @2048.
**Dự đoán:** % bị cắt giảm nhanh hơn GRPO-only (vd 25% → 15% ở cùng bước); Δ TB @512 **+1…+2**, @2048 ≈ 0 (nếu @2048 giảm → nén quá tay).
```
Lọc — r01_compress_sd — QUA (gần OPSD/SDFT; khác: chỉ dùng thầy để nén lời giải đúng, chọn mẫu theo độ dài)
G3 y* (thầy) + phần thưởng (chỉ lời giải đúng) cùng bước · G5 không chạm lời giải sai → ít rò; % cắt là chỉ số, không phải phần thưởng · G6 1,2× · G7 longonly=0 · G8 novel=0 = GRPO-only
```

## R01-5: Reward dày từ thầy, kiểu cộng, trên mọi câu — method `opsd --opt allfail_only=0`
_Thêm 2026-11-21 theo người dùng ("leverage thầy để reward dense")._
**Vấn đề:** P4 (một phần thưởng cho cả lời giải). **Cơ chế:** reward từng token `β·(log π_T − log π_S)` cộng vào phần thưởng cuối, trên mọi
lời giải; tương đương mục tiêu `E[r] − β·KL(π_S ‖ π_T)`, nghiệm tối ưu `π* ∝ π_T·exp(r/β)` — RL có ràng buộc KL về phía thầy-có-`y*`.
Khác R01-2 (kiểu nhân, không đổi mục tiêu) và `opsd` mặc định (chỉ câu cả-nhóm-sai). **Đối chứng:** `lam=1` = GRPO-only; `allfail_only=1` = `opsd`.
**Rủi ro:** imitation gap — thầy biết điều trò không biết → dạy đoán mò; rò đáp án. **Chi phí:** + 1 forward thầy trên mọi lời giải ≈ 1,3×.
**Dự đoán:** Δ vs novel=0 **0…+2**; % bị cắt giảm (thầy-có-Luna thiên về ngắn); KL thầy–trò giảm đều; nếu TB giảm mà KL giảm mạnh → đoán mò.
```
Lọc — opsd allfail_only=0 — QUA (≈ OPD/OPSD + GRPO; gần RLSD, JOLT+; ghép được) · G5 phần thưởng cuối giữ nguyên, β nhỏ · G7 lam=1, allfail_only=1 · G8 lam=1 → GRPO-only
```

## R01-6: Shaping dạng thế năng từ xác suất đáp án đúng — method `r01_potential`
_Thêm 2026-11-21._
**Vấn đề:** P4 + P1, không đổi mục tiêu (khác R01-5). **Cơ chế (lời thường):** cắt lời giải ở tối đa 4 chỗ xuống dòng; ở mỗi chỗ cắt, hỏi chính
model "nếu dừng ở đây và chốt `\boxed{`, xác suất viết ra đúng đáp án là bao nhiêu" = V. Đoạn lời giải làm V tăng được cộng advantage
`η·(V_k − V_{k−1})`, đoạn làm V giảm bị trừ. Shaping dạng thế năng không đổi chính sách tối ưu (Ng và cs. 1999) → không có đường ăn gian
phần thưởng. Câu cả-nhóm-sai vẫn có tín hiệu: đoạn nào đưa model lại gần đáp án đúng. Dùng đáp án (không dùng lời giải `y*`).
**Đối chứng:** `novel=0` = GRPO-only · `eta` (độ lớn). **Chi phí:** ≤ 5 forward ngắn mỗi lời giải (không grad) ≈ 1,3–1,5×. Chỉ số: V đầu/cuối
trung bình, tổng |dense| mỗi lời giải.
**Dự đoán:** Δ +1…+2,5; V cuối tăng theo bước nhanh hơn GRPO-only; tỉ lệ nhóm cả-8-sai giảm.
```
Lọc — r01_potential — QUA (gần VinePPO, PRIME/implicit PRM, RLPR/VeriFree (xác suất đáp án làm phần thưởng); khác: dùng làm thế năng → bảo
toàn nghiệm, chỉ cắt ở xuống dòng) · G3 đáp án đúng (thông tin giám sát) + phần thưởng cùng bước · G5 không đổi nghiệm tối ưu · G6 1,3–1,5× · G8 η=0 → GRPO-only
```

## Mốc chạy kèm (không phải ý tưởng)
GRPO-only (1 epoch, và 2 epoch cho đối chứng cùng ngân sách R01-1) · Guide (= `r01_guide_tokw --opt tokw=0`) · OPSD thuần (thầy có `y*`, không
GRPO) · SFT-Luna→GRPO · điểm neo phase 1 (SFT hint / self / Luna).
