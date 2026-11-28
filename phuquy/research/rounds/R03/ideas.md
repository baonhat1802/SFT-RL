# Vòng R03 — ý tưởng (cs-brainstorm), 2026-11-27
**Trọng tâm (người dùng, 27/11):** đưa causal inference / causal learning vào; "có cả tỉ phương pháp học, phương pháp phối hợp", không kết luận
"kín văn liệu".
**Thiết lập:** Qwen2.5-3B base, `curated`, `reward.py`, LoRA r32 α64, trần 2.048 token, 1 seed explore. Ý tưởng chỉ đem train (1 run so với
`grpo_s1` có sẵn) khi qua kiểm trước bậc nhất (`diag_cos.py`, không train).

## Bước 1 — vấn đề (số đo `runs/diag_ystar`, `runs/diag_wt`, `runs/diag_cos`, `runs/diag_fork`; model gốc)
| # | Vấn đề | Số đo |
|---|---|---|
| P1 | Văn bản y* trộn hai thứ: NỘI DUNG (phép tính, cách làm) và GIỌNG VĂN của Luna. SFT không tách được, và phần giọng văn chống phần thưởng | SFT-y* −0,15; chữ −0,13 (NLL 2,03/token), ký hiệu −0,12, số −0,06; SFT tự-thân +0,07; cos(SFT-y*, SFT-tự-thân) −0,12 |
| P2 | Gradient từ y* và gradient từ lời giải của chính trò cùng câu có phần chung (nội dung) và phần riêng (giọng); chưa có cách giữ phần chung | cos −0,12 tổng thể nhưng từng câu chưa đo |
| P3 | Tín hiệu thầy-có-y* đổi theo NGỮ CẢNH (prompt) hơn là theo y*: thầy phản ứng ~1% token; prompt ngắn gọn ≈ 0, prompt đầy đủ +0,15 | diag_cos, format-sweep |
| P4 | ~30% nhóm cả-sai, không gradient; 24% cứu được chỉ bằng thêm mẫu | diag_fork |

Ngôn ngữ nhân quả: trong y*, giọng văn là **biến gây nhiễu (confounder)** giữa "văn bản y*" và "học được gì". Muốn ước **hiệu ứng của nội dung**
phải chặn đường đi qua giọng văn: (a) can thiệp đổi nội dung mà giữ nguyên giọng (phản thực tế), (b) tìm phần bất biến qua các "môi trường"
khác giọng (học nhân quả kiểu IRM), (c) đối chứng âm (placebo) cho ngữ cảnh thầy.

---
## R03-1: Gradient nội dung bằng cặp phản thực tế cùng giọng — method `r03_cfpair`
Vấn đề: P1.
Cơ chế (lời thường): với mỗi y*, tạo bản sao **phản thực tế** ỹ*: giữ nguyên từng chữ và từng ký hiệu, chỉ đổi các giá trị tính ra (số trong
công thức) thành giá trị sai. Hai văn bản cùng giọng Luna, chỉ khác đúng/sai về nội dung. Tín hiệu học = tăng log p(y*) − log p(ỹ*): phần giọng
văn có mặt ở cả hai nên **triệt tiêu**, chỉ còn "chọn giá trị đúng thay vì giá trị sai trong cùng ngữ cảnh". Ví dụ: y* "x = (1−√3)³ = 10−6√3";
ỹ* "x = (1−√3)³ = 10−4√3" — model chỉ được dạy rằng 10−6√3 đúng hơn 10−4√3, không bị kéo về cách viết của Luna.
Dùng y* và phần thưởng: y* (và ỹ* sinh từ y*) cho số hạng tương phản; GRPO trên mẫu của trò như cũ; tổng = GRPO + λ·tương phản.
Vì sao không quy về GRPO-only hay SFT→GRPO: tắt tương phản → GRPO-only; thay ỹ* bằng rỗng → SFT-y* (đã đo −0,15). Phần mới là cặp phản thực tế
giữ giọng, đúng phép can thiệp do(nội dung) khi giọng cố định.
Đối chứng novel=0: λ=0. Ablation: `pert=numbers|operators|shuffle_steps`.
Cài đặt: +2 forward/backward trên y* và ỹ* (ngắn, ~115 token) mỗi câu; ~1,05× GRPO. Opts `lam=0.1, pert=numbers`. Log: margin log p(y*) − log p(ỹ*).
**Kiểm trước:** cos(∇NLL(y*) − ∇NLL(ỹ*), g_GRPO câu khác). Sống nếu ≥ +0,05 ở cả 2 batch và > SFT-y* (−0,15) rõ.
Dự đoán: kiểm trước +0,02…+0,08 (số tự nó −0,06, nhưng tương phản bỏ phần giọng); train Δ TB 0…+1.
```
Lọc cs-novelty-gate — r03_cfpair — QUA (trung bình)
G1 P1 −0,15 do giọng văn · đường nhân quả: cố định giọng, đổi nội dung → tín hiệu chỉ còn nội dung
G2 gần: dò rò rỉ bằng biến thể đổi số (2605.21856, chỉ đánh giá), tiền tố có một lỗi cài sẵn + GRPO (2512.17079, không có số hạng tương phản),
   SFT hỏng rồi GRPO (2611.26756), DPO/unlikelihood với mẫu âm tổng hợp · khác: mẫu âm là bản sao CÙNG GIỌNG của y* để triệt giọng, ghép GRPO
G3 y* + phần thưởng cùng bước · G4 không thầy ngoài · G5 nhị phân giữ nguyên · G6 ~1,05× · G7 λ=0 · G8 neo: pert=shuffle_steps phải ≤ numbers
```

## R03-2: Gradient bất biến qua "giọng" (học nhân quả kiểu IRM / AND-mask) — method `r03_invar`
Vấn đề: P1 + P2.
Cơ chế (lời thường): coi mỗi giọng văn là một **môi trường**: môi trường Luna (y*) và môi trường của chính trò (lời giải đúng nó tự sinh cho
cùng câu). Cơ chế nhân quả thật ("cách giải đúng") phải giống nhau ở mọi môi trường; giọng văn thì khác. Chỉ cập nhật những hướng mà gradient
của cả hai môi trường **cùng dấu** (AND-mask, Parascandolo 2020) — hướng chỉ một bên muốn (giọng Luna, hay thói quen riêng của trò) bị chặn.
Bản đầy đủ thêm môi trường thứ ba là gradient GRPO, để phần thưởng có quyền phủ quyết. Ví dụ: một hướng tham số làm tăng "Hence" (chỉ Luna
muốn) bị chặn; hướng làm tăng việc áp hằng đẳng thức u³+v³ (cả hai giọng đều dùng) được giữ.
Dùng y* và phần thưởng: y* = môi trường 1; lời giải đúng của trò (do phần thưởng lọc) = môi trường 2; GRPO = môi trường 3 / số hạng chính.
Vì sao không quy về GRPO-only hay SFT→GRPO: mask = 1 → SFT-y* + SFT-tự-thân + GRPO trộn thường; mask tắt hết → GRPO-only. Phần mới là chọn
hướng theo tính bất biến giữa giọng.
Đối chứng novel=0: mask tắt (chỉ GRPO). Ablation: `envs=ystar,own | ystar,own,grpo`, `mask=hard|soft`.
Cài đặt: +1 backward SFT-y*, +1 backward SFT-tự-thân mỗi rollout (đã có từ mẫu GRPO); ~1,15×. Log: tỉ lệ toạ độ qua mask, cos giữa các môi trường.
**Kiểm trước:** cos(mask⊙trung bình(g_y*, g_own), g_GRPO câu khác) so với g_own (+0,07); bản 3 môi trường so với mốc GRPO-vs-GRPO. Sống nếu bản
2 môi trường > g_own và bản 3 môi trường ≥ mốc ở cả 2 batch.
Dự đoán: 2 môi trường +0,05…+0,10; 3 môi trường ngang hoặc nhỉnh hơn mốc.
```
Lọc cs-novelty-gate — r03_invar — QUA (trung bình)
G1 P1/P2 cos(SFT-y*, SFT-tự-thân) −0,12 · đường nhân quả: giữ phần chung giữa hai giọng = nội dung
G2 gần: AND-mask (2009.00329), IRM; Invariant Gradient Alignment cho chưng cất suy luận (2606.05025: môi trường = nhóm bài đồng cấu, mask mềm,
   không có RL) · khác: môi trường = GIỌNG VĂN (y* vs lời giải tự sinh) + phần thưởng phủ quyết, trong vòng GRPO
G3 y* + phần thưởng cùng bước · G4 ok · G5 ok · G6 ~1,15× · G7 mask tắt = GRPO-only · G8 neo: mask ngẫu nhiên cùng mật độ phải kém mask thật
```

## Không thành thẻ (trùng) → prior_art.md
- Thầy trừ placebo (log p_T(·|x, y*) − log p_T(·|x, y* của câu khác)) — đối chứng âm cho ngữ cảnh (P3). Rất gần 2610.13744 (tương phản với placebo
  trong tự chưng cất có phê bình) và 2608.09228 (phá cặp mục tiêu–tham chiếu nhưng giữ ngữ cảnh). Vẫn đo kèm trong cùng lượt kiểm trước như một
  phép chẩn đoán P3 (miễn phí), không đem train.

---
## KẾT QUẢ KIỂM TRƯỚC (27/11, `runs/diag_causal`, 2 batch × 256 câu, mốc GRPO +0,074 / +0,087)
| Tín hiệu | batch 1 | batch 2 |
|---|---|---|
| R03-1 y* − đổi 1–2 giá trị | +0,096 | +0,114 |
| R03-1 y* − đổi dấu ± | +0,080 | +0,022 |
| đối chứng đảo dòng | +0,002 | −0,081 |
| SFT-y* | +0,025 | +0,070 |
| R03-2 AND 2 giọng / trung bình không mask / mask ngẫu nhiên | +0,037 / +0,048 / +0,034 | +0,151 / +0,137 / +0,105 |
| R03-2 AND 3 môi trường | +0,049 | +0,129 |
| placebo: thật / y* câu khác / hiệu | −0,003 / −0,002 / −0,008 | +0,123 / +0,095 / +0,056 |
**R03-1 `pert=numbers` QUA** (≥ +0,05 cả 2 batch, > SFT-y*, neo đảo dòng ≤). **R03-2 LOẠI** (mask ≈ trung bình thường). Placebo: tín hiệu thầy chủ
yếu do ngữ cảnh. Bước kế: cài `r03_cfpair` (λ=0,1, pert=numbers), train 1 run so với `grpo_s1` — chờ người dùng đồng ý.

---
## R03-4: Phần thưởng phụ theo KHÔNG GIAN NỘI DUNG học bằng cặp phản thực tế (CRL) — method `r03_crlsim`
Vấn đề: P1 (giọng văn che nội dung) + P4 (~30% nhóm cả-sai, không gradient).
Cơ chế (lời thường): học một đầu chiếu tuyến tính trên trạng thái ẩn của chính Qwen2.5-3B sao cho y* và lời giải ĐÚNG tự sinh của cùng câu (chung
nội dung, khác giọng) nằm gần nhau, còn y* và ỹ* (cùng giọng, sai 1–2 giá trị) hay lời giải của câu khác nằm xa — theo kết quả nhận dạng
nội dung/giọng của học tương phản trên cặp (von Kügelgen 2021, Locatello 2020). Trong GRPO, nhóm cả-sai nhận phần thưởng phụ β·độ gần nội dung
giữa lời giải của trò và y*, nên có gradient mà không bắt trò chép giọng Luna. Ví dụ: 8 lời giải sai của một bài tổ hợp; lời giải nào đã chia
trường hợp giống y* (dù viết kiểu của trò) được xếp trên lời giải đếm trùng.
Dùng y* và phần thưởng: y* = mỏ neo trong không gian nội dung; phần thưởng nhị phân lọc lời giải đúng làm cặp dương cho đầu chiếu và vẫn là
phần thưởng chính; phần phụ chỉ bật ở nhóm cả-sai.
Đối chứng novel=0: β=0 (GRPO-only). Cờ `enc=crl|untrained|lexical` (thay đầu chiếu học bằng embedding chưa train / chồng lấp từ — bản sau gần RL2R).
```
Lọc cs-novelty-gate — r03_crlsim — QUA (kèm mốc RL2R)
G1 P1 SFT-y* −0,09 TB 6 batch, phần chữ −0,13; P4 allfail 0,26–0,34 · đường nhân quả: phần thưởng phụ chỉ nhạy với nội dung → gradient ở nhóm cả-sai
   mà không kéo giọng văn
G2 gần: RL2R 2504.09895 (đọc phương pháp: BERTScore câu trả lời + từng bước so với y*, bổ sung cho kiểm đúng/sai, Qwen2.5-Math — encoder CÓ SẴN, nhạy
   bề mặt/giọng; ở đây encoder HỌC để bất biến giọng từ cặp phản thực tế), 2509.13081 (cosine encoder có sẵn, không phải toán), XIRL 2106.03911
   (encoder bất biến hình thái robot từ demo → phần thưởng khoảng cách; ở đây "hình thái" = giọng văn), Factored causal RM (tách z_c cho reward model
   RLHF, chống thiên lệch độ dài) · truy vấn: "reward embedding similarity reference GRPO style-invariant", "content style disentanglement contrastive
   paired texts identifiability", "causal representation learning RL reward shaping demonstration embedding distance invariant encoder"
   · ghép (RL2R + encoder nhân quả) — được theo luật nới 21/11; RL2R-kiểu (`enc=lexical/untrained`) thành MỐC
G3 y* (mỏ neo) + phần thưởng (lọc cặp dương, phần thưởng chính) trong cùng rollout · G4 chỉ Qwen2.5-3B (trạng thái ẩn của chính nó)
G5 nhị phân giữ nguyên, phần phụ ≤ β chỉ ở nhóm cả-sai (không bao giờ vượt lời giải đúng); gaming: kiểm AUC của độ dài làm mốc — nếu độ gần chủ yếu
   theo độ dài → loại · G6 ~1,05–1,1× (lấy trạng thái ẩn từ lượt tính log-prob cũ) · G7 β=0; dự đoán: allfail giảm nhanh hơn, Δ TB 0…+1
G8 neo: trên câu giữ ngoài, đầu chiếu phải xếp sim(y*, lời giải đúng) > sim(y*, ỹ*) (đúng mục tiêu train); `enc=untrained` ≈ mức của RL2R
```
**Kiểm trước (`code/beat/diag_crl.py`):** model gốc, 1.024 câu × 8 mẫu @2048; train đầu chiếu trên 768 câu, đo trên 256 câu giữ ngoài. Chỉ số: AUC
trong-nhóm (nhóm lẫn đúng/sai: độ gần với y* có xếp lời giải đúng trên lời giải sai không, loại được độ khó của câu). Mốc: embedding chưa train,
chồng lấp từ (kiểu RL2R), thầy-có-y* (log p_T − log p_S trung bình), độ dài. Sống nếu AUC(crl) ≥ 0,65 và hơn mọi mốc ≥ 0,05.
**27/11 — người dùng dừng hướng nội dung/giọng văn.** R03-4 không chạy (`code/beat/diag_crl.py` để lại, chưa dùng). Chuyển sang các góc nhân quả khác.
