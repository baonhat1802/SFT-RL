# Cơ chế đã có / đã chết / đã bị bác — tra trước khi đề xuất (skill cs-novelty-gate)
Mỗi dòng: cơ chế (mô tả việc nó làm) — trạng thái — nguồn. Thêm dòng mới ở cuối mục tương ứng.
Mô tả ở mục B lấy từ `related-work/README.md` (phần lớn từ tóm tắt, chưa đọc toàn văn) — đọc phần phương pháp trước khi kết luận trùng/không trùng.

## A. Người dùng đã bác (không đề xuất lại dưới bất kỳ dạng nào)
_(chưa có)_

## B. Đã có trong văn liệu
### Gộp SFT + RL
- SFT rồi RL hai giai đoạn — baseline; SFT-then-RL (arXiv 2604.23747, NeurIPS 2026): khi cài đúng (sửa lỗi cộng dồn gradient, optimizer),
  các phương pháp gộp không thắng SFT→RL **trong setup của họ** (lời giải R1 dài).
- Trộn lời giải mẫu vào nhóm GRPO, importance sampling có uốn `p/(p+0,1)` — LUFFY (2504.14945).
- Loss SFT + loss RL trong cùng batch, trọng số theo entropy — SRFT (2506.19767).
- SFT là loss phụ trong vòng RL on-policy, hệ số chung μ + trọng số từng token — CHORD (2508.11408).
- Học trọng số SFT/RL bằng meta-gradient — AMFT (2508.06944).
- SFT và RL là hai trường hợp của một bài tối ưu, chuyển qua lại theo kết quả rollout — HPT (2509.04419).
- SFT chen vào RL chỉ trên câu khó (cả nhóm sai) — ReLIFT; theo tiền tố lời giải — Prefix-RFT; TRAPO.
- **Hai tầng leader–follower: LoRA thầy `w` (tầng trên, mục tiêu SFT log-likelihood trên `y*`) + trò `θ` (tầng dưới, RL), policy
  `π_{θ+w}`, giải bằng phạt `p = max_θ' J_RL − J_RL(θ)` → max–min với trò bóng chỉ-RL; trò cập nhật tổ hợp lồi gradient SFT/RL; thầy cập
  nhật bậc một qua một bước RL ảo θ̂** — BRIDGE (2509.06948, ICML 2026, §3). Dữ liệu SFT: lời giải R1 chưng cất.
### Chưng cất on-policy (thầy là model khác) — vi phạm G4 trên capstone nếu cần thầy ngoài
- OPD reverse KL từng token (Thinking Machines blog); GKD (2306.13649); KDRL; MOPD (2606.30406); TREK (2607.05339: câu khó trò không giải
  ra → lời giải thầy *hoặc chính nó có thêm ngữ cảnh*, forward-KL ngắn, rồi về GRPO); weak-to-strong OPD (2607.05394).
### Tự chưng cất on-policy (thầy = chính model + thông tin đặc quyền) — họ sát hướng capstone nhất, tra kỹ
- Thầy có lời giải mẫu trong ngữ cảnh — SDFT (2601.19897); OPSD / Self-Distilled Reasoner (2601.18734).
- Thầy có phản hồi môi trường — SDPO (2601.20802); ghép SDPO + GRPO — SRPO (2604.02288).
- Phần thưởng quyết định chiều, thầy quyết định độ lớn từng token — RLSD (2604.03128).
- Thầy xem một lần thử thành công trong cùng nhóm — HSD (2606.15576); thầy dừng gradient chỉnh advantage từng token — CAST (2606.18810);
  H²SD (2607.18955); OGLS-SD (2605.12400); entropy thầy — 2605.13255; DemoPSD (2607.02502); UniSD (2605.06597); π-Play (2604.14054);
  Rebellious Student (2605.10781); SD-Search, AHEAD, GRSD.
### Ngoài LLM (kiểm khi ý tưởng có thầy học theo phản hồi của trò)
- Thầy sinh nhãn giả, được cập nhật theo mức trò tiến bộ trên dữ liệu có nhãn — Meta Pseudo Labels (Pham et al., CVPR 2021).
- Học với thông tin đặc quyền (LUPI, Vapnik & Vashist 2009); chưng cất đặc quyền (privileged distillation).
- STaR "rationalization": model xem đáp án rồi viết lại lời giải, train trên đề không có đáp án (Zelikman et al. 2022) — **đây chính là
  `gen_sft_data --mode hint`**, nên bản tách rời của nó chỉ là điểm neo, không phải ý tưởng.

### Gợi ý cho câu khó trong lúc RL (tìm 2026-11-21, R01)
- Gợi ý vào prompt chỉ cho câu cả nhóm sai + hiệu chỉnh trọng số quan trọng về chính sách không gợi ý — Guide (2506.13923); OC-GRPO
  (NeurIPS 2026, "off-context", đổi độ đo); "Do Not Step Into the Same River Twice" (2510.26109, tỉ lệ chuẩn hoá theo LUFFY).
- Model gợi ý sinh hint từ lần thử hỏng + lời giải mẫu, chọn hint theo việc nó giúp chính sách không gợi ý — HiLL (2604.00698).
- Gợi ý nối vào prompt, giữ on-policy — SAGE (2602.03143). Thay một lời giải bằng đáp án vàng khi cả nhóm sai — GCPO (2510.07790).
### Thầy tự chưng cất được train (tìm 2026-11-21, R01)
- Một chính sách hai vai: thầy (có ngữ cảnh đặc quyền) học bằng phần thưởng của chính nó + KL về trò; trò học OPSD; JOLT+ thêm phần thưởng
  cho trò; không hai tầng, không tín hiệu lợi ích của trò — JOLT (2610.10447, 10/2026; MATH +4,2 so với GRPO).
- Thầy dạy kèm được train bằng RL theo kết quả học của trò mô phỏng — Sherpa (2610.08778; dạy kèm nhiều lượt, không phải chưng cất).
- Thầy có lời giải mẫu ép đổi hành vi ở mọi token; phê bình theo bước tốt hơn — 2606.11173. Đảo chiều chưng cất trên lời giải đúng — RLRT
  (2605.10781). Chưng cất "bài học" thay vì cả lời giải — ROSD (2605.28014). Tự gợi ý — HDPO, π-Distill.

## C. Đã chết / đã đo trong project
- SFT trên lời giải Luna (LoRA r32, lr 1e-4, 3 ep): 14,4 → 8,1 (bảng nhóm) — lời giải quá ngắn, model 3B bắt chước kiểu viết mà không theo
  kịp lập luận (giả thuyết; phase 1 đang kiểm bằng SFT trên dữ liệu tự viết).
- SFT-Luna → GRPO (LoRA và full FT): 17,0–17,2 < GRPO-only 21,9–22,3; bị cắt 42–55% (bảng nhóm).

### Tìm 2026-11-26 (R02) — đều LOẠI vì trùng
- Thầy tương phản (có lời giải đúng trừ có lời giải sai) — RLCSD (2606.11709), "Repulsive and Attractive Teachers" (2609.21561), Anti-Self-
  Distillation PMI (2605.11609), SSOPD (2605.17497: thầy có lời giải đúng ngắn nhất, chưng cất lên tiền tố lời giải sai dài nhất).
- Thầy có bộ nhớ lời giải đúng cũ của chính model — Procedural Memory Distillation (2607.01480); thầy có lần thử thành công — RLRT, SDPO;
  replay lời giải đúng — ExGRPO (2510.02245).
- Đồng thuận nhiều thầy / tách phần chung–phần riêng — AVSD (2605.20643), CANON (2607.13643), GATES (2602.20574), multi-teacher adaptive
  OPSD (2610.40574). Ý "ADVISOR cho LLM" (Weihs 2021: bắt chước nơi bắt chước được, RL nơi cần thông tin đặc quyền) quy về nhóm này.
- Câu hỏi con kiểm chứng được tách từ lời giải mẫu — SCRL (2605.22074, thầy ngoài sinh 4 câu con), VERIFY-RL; R³ (2402.05808).
- RLSD trọng số từng token — đã có (2604.03128); đo 26/11: không thêm tín hiệu tại model gốc (diag_tokw).
- SDPG (gate dương + lịch hệ số chưng cất), Off-Context GRPO (2607.19313: ngữ cảnh đặc quyền điều khiển phân phối rollout).

## D. Đo trong project (26/11)
- cos(g_Q,g_S) trong log R01 là ảo do đo tại θ′ — bước theo gradient GRPO cũng tụt ~0,25 khi đo tại θ′ (`runs/diag_cos`, 3/3 batch).
- Rẽ nhánh trên lời giải sai tại chỗ thầy-có-y* phản đối (R02-1 `r02_fork`) — CHẾT ở kiểm trước 26/11: không hơn sinh lại từ đầu (21,4% vs
  23,7% câu cả-sai được cứu), kém vị trí ngẫu nhiên cùng độ sâu (18,3% vs 24,0%). Rẽ sâu (random 15%, entropy 11%) còn kém hơn — với Qwen2.5-3B,
  tiền tố của lời giải sai làm giảm cơ hội đúng. `runs/diag_fork`, `code/beat/diag_fork.py`.
- ~24% câu cả-8-sai được cứu bằng 8 mẫu nữa từ đầu → phân bổ thêm rollout cho câu cả-sai có tác dụng, nhưng đã có (adaptive rollout
  allocation / Knapsack RL) — không phải ý tưởng mới.

### Sửa sau khi đọc kỹ (2026-11-26, `related-work/reading_2026-11-26.md`)
- Các phán quyết "trùng" ở mục R02 trên đây dựa vào tóm tắt tìm kiếm; đọc toàn văn xác nhận CƠ CHẾ trùng (thầy tương phản, nhiều thầy,
  bộ nhớ đều có thật), nhưng **tất cả dùng Qwen3/Olmo instruct hoặc thinking, ngân sách 4k–38k**. Model base + toán + ngân sách ngắn —
  setup capstone — chỉ có BRIDGE (thắng, nhưng thầy là LoRA SFT trên vết R1, không phải tự chưng cất), SCRL (thầy ngoài), VPD (SDPO sụp 0%).
- P2S không phải toán (DROP, MedicalQA) và chỉ cần đáp án; SCRL dùng LLM ngoài (G4 không cho), bản tự sinh chỉ là ablation.
- AntiSD (2605.11609): ĐẢO DẤU tín hiệu chưng cất mới thắng GRPO (65,7 vs 57,4); chưng cất đúng chiều 30,6.
- BRIDGE (2509.06948): người dùng đánh giá 26/11 là có nhiều mâu thuẫn, có vẻ AI slop dù qua bình duyệt → không dùng làm bằng chứng
  "SFT+RL kết hợp thắng trên model base", không làm mốc số, không phân tích tín hiệu thầy của nó như đóng góp.

- Thầy = model sau một bước SFT trên y* của từng câu (y* trong trọng số thay vì ngữ cảnh), R02-3 — CHẾT ở kiểm trước 27/11: chuyển giao âm ở
  mọi độ lớn bước (−0,16…−0,22) dù phản ứng mạnh; gradient SFT trên y* tự nó −0,18 (`runs/diag_wt`).

### Tìm 2026-11-27 (R03, nhân quả)
- Thầy trừ placebo / phá cặp tham chiếu — 2610.13744 (placebo contrast, tự chưng cất có phê bình), 2608.09228 (mismatched pairing). Chỉ đo chẩn đoán.
- AND-mask / gradient bất biến trong chưng cất suy luận, môi trường = nhóm bài đồng cấu — Invariant Gradient Alignment (2606.05025). R03-2 khác môi trường.
- Lệch giọng văn khi bắt chước văn bản có tham chiếu — DASD (2606.00628), 2603.24472. Khớp chẩn đoán `runs/diag_ystar`.
- Cặp phản thực tế/đổi số — 2605.21856 (dò rò rỉ), 2512.17079 (tiền tố có lỗi + GRPO), 2611.26756 (SFT hỏng rồi GRPO), 2601.07408 (credit phản thực tế).
- AND-mask/gradient bất biến giữa giọng y* và giọng của trò (R03-2 `r03_invar`) — CHẾT ở kiểm trước 27/11: mask cùng dấu ≈ cộng hai gradient
  thường (+0,094 vs +0,093), 3 môi trường dưới mốc ở batch 1 (`runs/diag_causal`).
- Thầy-ngữ-cảnh: thầy với y* của CÂU KHÁC cho tín hiệu gần bằng thầy thật (+0,046 vs +0,06) → phần lớn tín hiệu là do ngữ cảnh (`runs/diag_causal`).
