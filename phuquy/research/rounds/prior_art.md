# Cơ chế đã có / đã chết / đã bị bác — tra trước khi đề xuất (skill cs-novelty-gate)
Mỗi dòng: cơ chế (mô tả việc nó làm) — trạng thái — nguồn. Thêm dòng mới ở cuối mục tương ứng.
Mô tả ở mục B lấy từ `related-work/README.md` (phần lớn từ tóm tắt, chưa đọc toàn văn) — đọc phần phương pháp trước khi kết luận trùng/không trùng.

## A. Người dùng đã bác (không đề xuất lại dưới bất kỳ dạng nào)
- Hướng "tách nội dung / giọng văn của y*" (27/11): người dùng thấy bị lậm vào khung này, dừng. Gồm R03-4 `r03_crlsim` (không gian nội dung
  học bằng cặp phản thực tế, CRL) — chưa chạy kiểm trước; R03-2 đã chết. R03-1 `cfpair_s1` đang chạy thì chạy hết làm mốc, không mở rộng thêm.

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
### Tìm 2026-11-27 (sau khi dừng hướng nội dung/giọng văn)
- Độ dài là biến gây nhiễu của phần thưởng / thiên lệch độ dài trong GRPO — Dr. GRPO (2503.20783), LUSPO (2602.05261), GRPO-LEAD (2504.09696). Không dùng y*.
- Thiên lệch theo độ khó / đặt lại trọng số theo tỉ lệ giải — HA-DW "Your Group-Relative Advantage Is Biased" (2601.08521), GRPO-LEAD, 2607.00152. Không dùng y*.
- Chọn phần gợi ý từ lời giải theo tác động lên tỉ lệ đúng (bỏ-từng-phần, tập tối thiểu đủ) — KnowRL (2604.12627: điểm kiến thức do DeepSeek-R1
  trích, chọn tập con offline bằng độ chính xác khi bỏ từng điểm); gợi ý tiền tố lời giải — QuestA (2507.13266), ADHint (2512.13095); model tự dùng y*
  tìm lỗi đầu tiên rồi thay một bước — InT (2601.14209).
- Cặp phản thực tế cùng giọng, dạng tuyến tính λ·[NLL(y*) − NLL(ỹ*)] (R03-1 `cfpair`) — kiểm trước QUA (+0,10/+0,11) nhưng train SỤP từ rollout ~20
  (`experiments/rounds/R03/cfpair_s1`): margin tăng không chặn. Bản có chặn chưa thử.
- Thầy-có-y* làm control variate phụ thuộc hành động, bù chính xác trên từ vựng (AIPW/doubly robust cho GRPO, R04-1) — CHẾT ở kiểm trước 27/11:
  |cos(CV, GRPO)| ≈ 0,05, đổi dấu giữa batch → giảm nhiễu < 0,5% (`experiments/rounds/R04/diag_dr.log`).
- Lời nhắc chung tự đặt (nhắc lại đề / liệt kê định lý / trường hợp riêng / đi ngược) + chọn theo CATE (R04-3) — CHẾT ở kiểm trước 27/11: không nhánh
  nào hơn lấy mẫu thường (−0,01…−0,02), chọn theo câu kém chọn chung. Câu đầu của y* làm gợi ý: +0,065 (KTC 90% +0,045…+0,085) — gần QuestA/KnowRL.

## Rà văn liệu 27/11 (tối) — gợi ý/trừu tượng hoá cho câu khó, và RL thật ra mua được gì
- **HINT** (arXiv 2510.09388, Findings ACL 2026): meta-hint (gợi ý trừu tượng, không lộ đáp án) > answer-hint (LUFFY, BREAD) trong GRPO. Khớp
  `diag_insight`: gợi ý từng bước của Luna chủ yếu lộ giá trị; che số thì gần hết tác dụng.
- **NuRL** (arXiv 2509.25666): tự sinh gợi ý trừu tượng cho câu 0% đúng; bật gợi ý SAU khi GRPO ổn định tốt hơn bật từ đầu.
- **RLAD** (arXiv 2510.02263, ICLR 2026): hai người chơi — bộ sinh trừu tượng được thưởng bằng mức tăng độ đúng của bộ giải; khởi động bằng
  trừu tượng từ o4-mini. Gần nhất với ý "chọn gợi ý theo hiệu ứng can thiệp lên trò" (HiLL cũng thưởng hinter theo mức tăng của trò).
- **POPE** (arXiv 2601.18779), **QuestA** (2507.13266), **Off-Context GRPO** (2607.19313), **KnowRL** (2604.12627), **SAGE** (2602.03143),
  **HiLL** (2604.00698): tiền tố/gợi ý từ lời giải mẫu cho câu cả-sai — đã có trong mục trước, nhắc lại vì cùng họ.
- **ReasonFlux** (2502.06772) / **Buffer of Thoughts** (2406.04271): thư viện khuôn mẫu chiến lược rút từ lời giải.
- **What Outcome RL Buys** (arXiv 2610.34894): ở trần 4k–8k, 82% lợi ích pass@1 của RL là "viết xong trong ngân sách" (finishing), "với tới
  câu mới" (reach) ≤ 2,1 điểm. Khớp: GRPO@512 ≈ base@2048; `cfpair_reg` +4 TB @512 nhưng −1,5 @2048; khảo sát đọc tay (1/3 lỗi là hình thức).

## Rà văn liệu 27/11 (khuya) — khung "imitation gap / thầy có thông tin riêng" cho SFT+RL
- **ADVISOR** (Weihs và cs., NeurIPS 2021, arXiv 2007.12173): imitation gap; trộn IL/RL theo từng trạng thái, chỉ bắt chước nơi khoảng cách nhỏ
  (ước bằng một chính sách phụ chỉ thấy quan sát của trò). Robot/agent; tác giả nói chưa kiểm ở ngôn ngữ.
- **CHORD** (arXiv 2508.11408, ICLR 2026): SFT là số hạng phụ trong RL, μ toàn cục + trọng số token ϕ giảm token lệch nhiều (theo xác suất).
- **SRFT** (2506.19767): trọng số theo entropy. **AMFT** (2508.06944): meta-học μ toàn cục.
- **SGPO** (2606.24064): chưng cất CHIẾN LƯỢC thay vì quỹ đạo; trọng số từng câu α(q) theo khoảng cách pass-rate có-hướng-dẫn vs tự giải —
  gần nhất với "trọng số theo khả năng tái tạo", nhưng ở cấp câu, không ở cấp bước.
- **Small Models Struggle to Learn from Strong Reasoners** (2502.12143, ACL Findings 2025): learnability gap ở ≤3B; Mix Distillation.
- Viết lại dữ liệu thầy về phân phối trò: SDFT (Yang 2024), TESSY (2604.14164), student-conditioned trace (2610.23831), BSD (2610.26397).
- **Privileged AI feedback cho agent LLM** (2410.05434).
→ Khung tổng quát ĐÃ được công nhận; ô trống còn lại: (1) ước lượng khả năng tái tạo bằng CAN THIỆP (trò viết tiếp từ tiền tố thầy có đúng không)
  thay vì xác suất/entropy — có bằng chứng của mình rằng log-xác suất không đo được điều này (φ ≈ placebo; bước ý then chốt NLL thấp);
  (2) tách hình thức (tái tạo được) khỏi nội dung (không) theo đúng tiêu chí imitation gap, ở ngân sách token chặt.

## Rà văn liệu 28/11 — cfpair (cặp phản thực tế đổi giá trị) và "bộ kiểm từ lời giải mẫu" có ai làm chưa
- **RISE** (arXiv 2410.06638, ACL 2025): model tự sửa vài token then chốt trong lời giải ĐÚNG CỦA CHÍNH NÓ để cài lỗi tinh vi → cặp DPO
  (đúng vs bị cài lỗi) + cặp tự nhiên. GẦN NHẤT với cfpair. Khác: (1) ta cài lỗi vào LỜI GIẢI MẪU của thầy (y*), quy tắc đổi giá trị tính ra,
  giữ nguyên văn; (2) ta trộn ONLINE trong GRPO (số hạng phụ có chặn), RISE là DPO offline; (3) ta có ablation chỉ ra cặp TỰ NHIÊN và chỉ-kéo-y*
  đều sụp, placebo chỉ lấy ~1/3 lợi thế; (4) khung nhân quả (mang sang cơ chế kiểm, không mang sang chính sách) + đo bộ kiểm mang sang.
- **SCDPO** (2407.00782): giữ các bước đúng tới điểm k rồi cho model sinh tiếp để có bước sai → cặp DPO. **Step-DPO** (2406.18629): bước chọn/loại.
- **Hard Negative DPO cho model nhỏ** (2512.19728): lỗi cài vào (khác ít so với lời giải đúng) là mẫu âm khó.
- PRM: **Controllable & Verifiable Process Data Synthesis** (2605.02395) cài lỗi vào chuỗi logic ký hiệu rồi tính lại các bước sau (chuyển sang
  toán mới sơ bộ); **CMI PRM** (2604.10660) dùng lời giải mẫu GSM8K/MATH, không rollout, nhưng chấm bước bằng tương phản xác suất, không cài lỗi;
  **RAWR** (2603.17815), **Adversarial PRM** (2511.22888) sinh lỗi bằng generator đối kháng.
→ Kết luận: "cài lỗi tối thiểu → cặp tương phản" ĐÃ CÓ (RISE, hard negatives). Hướng B (PRM từ lời giải mẫu) dễ trùng, không ưu tiên.
  Phần riêng còn lại của mình: online trong RL + lời giải của THẦY mạnh hơn (không phải lời giải của chính model) + ablation nhân quả cho thấy
  CHỈ cặp giữ-văn-đổi-giá-trị mới chuyển được tín hiệu từ thầy, + phát hiện khoảng cách thầy–trò (phần nội dung ngược chiều khi trò đủ mạnh).
