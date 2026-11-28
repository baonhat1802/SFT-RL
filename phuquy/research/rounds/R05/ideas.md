# R05 — causal discovery / causal representation learning (27/11)

Người dùng (27/11): can thiệp vào biến ẩn / cấu trúc nhân quả, không chỉ can thiệp đầu vào; CRL gồm cả khám phá nhân quả, lệch phân phối,
cơ chế mô-đun. Thiết lập mặc định như R03/R04 (Qwen2.5-3B base, curated, reward.py nhị phân, LoRA r32 α64, 2048 token).

## R05-A: khung nhân quả của y* bằng khám phá đồ thị qua can thiệp — LOẠI ở kiểm trước
Ý: xoá từng bước của y*, đo độ tụt log-xác suất của các SỐ trong công thức ở các bước sau và của vùng đáp án → đồ thị bước → bước;
khung = tổ tiên của nút đáp án. Nếu SFT trên y* hại vì phần không nhân quả (giọng văn), SFT chỉ trên khung sẽ cùng chiều GRPO.
(Bản đầu đo trên MỌI token → đồ thị dày, đo độ trôi chảy chứ không phải phụ thuộc; đã đổi sang token nội dung.)
Kiểm trước (`diag_cos --skeleton-sweep`, 6 batch × ~60 câu, cos với gradient GRPO nửa B; `experiments/rounds/R05/skel_results.json`):
| SFT trên … | cos (TB [min; max]) |
|---|---|
| mốc GRPO–GRPO | +0,085 [+0,048; +0,126] |
| cả y* | −0,077 [−0,158; +0,012] |
| chỉ vùng đáp án | +0,026 [−0,034; +0,063] |
| khung τ=0,5 (32% bước, gồm đáp án) | −0,021 [−0,098; +0,025] |
| khung τ=0,5 bỏ đáp án | −0,065 [−0,168; +0,015] |
| đối chứng: bước ngẫu nhiên cùng số lượng + đáp án | −0,049 [−0,127; +0,011] |
| phần ngoài khung τ=0,5 | −0,093 |
| văn bản y* đã xoá bước ngoài khung | −0,080 |
| khung τ=0,1 (77% bước) | −0,072 (ngẫu nhiên −0,080) |
Đồ thị lọc đúng: chữ thường chỉ 2,5% trong khung τ=0,5 so với 54% ngoài khung. NHƯNG khung bỏ đáp án (−0,065) không hơn bước ngẫu nhiên
(−0,049); phần nhỉnh của khung có đáp án đến từ vùng đáp án (+0,026). → **Các bước TÍNH TOÁN của Luna cũng ngược chiều GRPO**, không riêng giọng
văn. Sửa lại cách đọc ở mentor_updates mục 6: "giọng văn" không phải toàn bộ câu chuyện; khớp với `diag_mediation` (lời giải có-y* chứa bước
nhảy model không tự theo được). Hướng "lọc y* theo cấu trúc nhân quả rồi SFT" chết.

## Khảo sát đọc tay: lời giải của model cạnh lời giải Luna (`diag_survey`, 27/11; `experiments/rounds/R05/survey.md`)
300 câu × 8 mẫu (TB đúng 0,273; cả-sai 107, khó 1–3/8 107, trung bình 85). Đọc 40 câu: 15 cả-sai, 15 khó, 10 trung bình.
1. **Khi model giải đúng câu khó, nó đi CÙNG đường với Luna** (Q16, 19–24: cùng phép biến đổi, cùng công thức), chỉ dài gấp ~4 (trung vị
   1.250 vs 329 ký tự): Luna bỏ phép tính số học và lời biện minh ("biệt thức 36" không ghi 900−864). Luna ≈ bản NÉN của chính đường của model.
   "Luna đi đường khác" hiếm trong mẫu này.
2. **Câu cả-sai: thiếu MỘT ý then chốt về cấu trúc mà Luna nói ngay ở 1–2 bước đầu**: g² | 180 (Q1), U+2V = 3x² (Q5), xen kẽ giới tính ⇒
   vị trí cố định ⇒ 3!·4! (Q8), khoảng cách tới hai cạnh bên tỉ lệ với AP (Q9), 24 | n² ⇒ 12 | n (Q10), hai hàng trực giao cùng độ dài (Q14).
   Model thay vào đó làm vét cạn (khai triển bậc 4, đệ quy/DP, liệt kê ước) rồi lạc. Câu mở đầu của model là chép lại đề.
3. **~1/3 lời giải sai ở câu khó/cả-sai, 76% ở câu trung bình là hỏng HÌNH THỨC chứ không phải lập luận**: không có `\boxed` (77/246 mẫu sai),
   trôi sang chữ Trung/Ba Tư/rác ở cuối, token rác đầu bài ("cretilliseconds", "czą", "/gif", ".urls➡️"), dừng ở code Python. Có mẫu tính ra
   đúng 4 ở câu "cả-sai" Q1 nhưng không đóng hộp → 0 điểm. GRPO sửa nhanh phần này — khớp với "phần lớn lợi ích GRPO là viết gọn/đúng thể thức".
4. **Độ bất ngờ (NLL) theo bước KHÔNG chỉ ra ý then chốt**: bước NLL cao là từ nối ("Hence", "There are", "Therefore the answer is"); bước ý
   then chốt chỉ ~1,2 vì khi ĐỌC thì ý đó hợp lý. Log-xác suất teacher-forcing đo độ hợp lý khi đọc, không đo "tự nghĩ ra được không" — lý do
   các phương pháp dựa trên log-ratio/xoá-bước (φ, khung nhân quả) không bắt được cơ chế.

## Cấy trạng thái ẩn (`diag_latent`, 27/11; 80 câu khó × 2 mẫu, trần 1024; `experiments/rounds/R05/latent_results.json`)
Trò (không y*) tự sinh từng token; ở mỗi bước, trạng thái ẩn sau lớp L tại vị trí hiện tại được thay bằng của bản sao CÓ y* đọc cùng chuỗi token.
| cấu hình | đúng | Δ [KTC 90%] | phần trung gian |
|---|---|---|---|
| không cấy | 0,150 | — | — |
| cả trạng thái, lớp 12 | 0,094 | −0,056 [−0,119; +0,006] | −13% |
| cả trạng thái, lớp 24 | 0,225 | +0,075 [+0,006; +0,144] | 17% |
| lớp 36 (cuối) = model có-y* (neo) | 0,600 | +0,450 | 100% |
| 64 hướng chính, lớp 24 (giải thích 44% phương sai hiệu) | 0,175 | +0,025 [−0,032; +0,081] | 6% |
| placebo: y* câu khác, lớp 24 | 0,119 | −0,031 | −7% |
Đọc: trên token của chính trò, thông tin riêng của y* (placebo ≈ 0) mới có ~17% trong dòng dư ở lớp 24 và ~0 ở lớp 12; phần lớn được đọc
thẳng từ ngữ cảnh ở 12 lớp cuối (chép). Không nằm trong không gian con hạng thấp (64 hướng: 6%). → không có "biến lập luận ẩn" gọn để chưng cất;
khớp với `diag_mediation` (văn bản có-y* không tự đứng được) và φ (thông tin riêng của y* ở vùng đáp án). Cỡ mẫu nhỏ, KTC rộng.

## Khám phá nhân quả bằng can thiệp: bước then chốt của y* (`diag_insight`, 27/11; 160 câu khó, 94 cả-sai; `experiments/rounds/R05/insight_results.json`)
Mỗi bước của Luna = một can thiệp do(gợi ý := bước k); đo tỉ lệ model TỰ giải đúng (8 mẫu). Mốc: không gợi ý 0,114; cả y* 0,727.
| vị trí bước | Δ một bước [KTC 90%] | Δ bước đã che số (`_`) | Δ cộng dồn bước 0..k | tỉ lệ lộ đáp án |
|---|---|---|---|---|
| 0 | +0,050 [+0,023; +0,077] | +0,041 | — | 6% |
| 1 | +0,058 | +0,012 | +0,150 | 16% |
| 2 | +0,035 | +0,038 | +0,212 | 23% |
| 3 | +0,045 | +0,001 | +0,248 | 25% |
| 4 | +0,047 | +0,021 | +0,303 | 33% |
| 5 | +0,068 | +0,031 | +0,296 | 29% |
Bước tốt nhất mỗi câu (chọn trên 4 mẫu, chấm trên 4 mẫu khác): 0,227 vs không gợi ý 0,105. 46% câu có một bước tăng ≥ 0,25; 18% ≥ 0,5.
(Biểu đồ vị trí bước tốt nhất 96/153 ở vị trí 0 KHÔNG dùng được: khi hoà thì chọn bước nhỏ nhất.)
Đọc:
1. **Không có "một bước then chốt ở đầu" theo trung bình**: mỗi bước đơn chỉ +0,04–0,07, vị trí đầu không trội; cộng dồn tăng đều (giống `diag_mediation`).
2. **Phần lớn hiệu ứng là GIÁ TRỊ, không phải Ý**: che số đi thì hiệu ứng tụt về +0,00–0,04. Ví dụ bước "mạnh nhất" thường chứa đáp án/giá trị
   gần đáp án mà bộ kiểm lộ (so chuỗi) bỏ sót: "M = (1 2; −3 −5)" (chính đáp án), "m=1, n=3", "r=10", "a=3".
3. **Ý then chốt thật có tồn tại nhưng hiếm**: vài bước giữ hiệu ứng khi che số — "đặt XZ = t, XY = 7t" (+0,75 → +0,50), "r√2 = 3 − r"
   (+0,75 → +0,75), tổng dãy = Σ C(m,j) = 2^m (+0,62 → +0,75). Đây là các bước ĐỔI CÁCH BIỂU DIỄN, khớp khảo sát đọc tay.
Kết luận: giả thuyết khảo sát (một ý then chốt ở đầu) chỉ đúng cho một phần nhỏ câu; tín hiệu gợi ý từng bước chủ yếu là lộ giá trị trung gian.

## Tìm chỗ có phương sai lớn (27/11 tối; từ generations của `cfpair_reg_s1` @512 và @2048, không chạy model)
- 1.517 câu eval: giải được ở 512 38,7% · **chỉ giải được khi có 2048: 16,0%** · không giải được cả hai 45,4%.
- Khoảng cách 512→2048 (TB): cfpair_reg 21,6 → 30,9 (+9,3); grpo_s1 17,6 → 32,4 (+14,8). Lớn nhất ở MATH500 (+14,6 điểm), Olympiad (+12,0), AMC (+11,3).
- Độ dài lời giải ĐÚNG @2048 (ước token): trung vị 436; ≤512: 67%; ≤640: 84%; ≤768: 92% → nén ~20–25% đủ đưa thêm ~17% lời giải đúng vào trần 512.
- Lời giải bị cắt @512 KHÔNG phải rác: lặp dòng 1%, lặp n-gram 0,7%, code 0,5%, chữ ngoại 0%; có "verify/check" 14%.
- Thí nghiệm tự nhiên: 144 câu có cả lời giải đúng NGẮN (≤512) và DÀI (>512). Cùng một đường giải; bản dài +25% token do chia nhỏ biến
  đổi (khai triển từng hạng tử thành khối riêng), lặp lại điều hiển nhiên ("hệ số t⁴: 12 = 12"), thêm tiêu đề/markdown (9% ký tự vs 0%),
  nhiều dấu "=" hơn (17 vs 12). → độ dài là ĐỘ CHI TIẾT của cùng một đường, không phải đường khác hay lặp vô ích.
- Huấn luyện: mỗi rollout ~30% nhóm cả-đúng + ~20% nhóm cả-sai → ~50% nhóm không có gradient GRPO.

## R05-B: "Transport the verifier, not the generator" — kiểm trước ĐẠT (`diag_verifier`, 27/11; `experiments/rounds/R05/verifier_results.json`)
Lập luận (transportability + ICM): cơ chế SINH bước giải phụ thuộc người giải (Luna có U) → không mang sang; cơ chế KIỂM tính hợp lệ bất biến
giữa người giải → mang sang được. Bộ kiểm = model gốc + LoRA r16, học CHỈ từ (y*, Đúng) / (ỹ*, Sai) trên 2.760 câu; thử trên lời giải của
chính model (256 câu khác, 8 mẫu; 163 nhóm lẫn; 75% lời giải viết xong).
| bộ kiểm | AUC trong nhóm (tất cả) | AUC trong nhóm (viết xong) | AUC gộp (viết xong) | AUC y*/ỹ* giữ lại |
|---|---|---|---|---|
| zero-shot | 0,711 | 0,663 | 0,677 | 0,594 |
| **học từ y*/ỹ*** | **0,763** | **0,738** | **0,816** | 0,891 |
| placebo (nhãn xáo) | 0,543 | 0,501 | 0,486 | 0,527 |
| mốc −độ dài | 0,492 | 0,519 | — | — |
(φ của thầy ở R04: 0,66, placebo φ 0,64.) Đọc: chỉ học từ lời giải Luna mà chấm đúng lời giải của chính model tốt hơn zero-shot +0,05…+0,14;
placebo về 0,5 → phần tăng là học tính hợp lệ, không phải quen định dạng. Kèm ablation `cfpair_pos_s1` (bỏ ỹ* → sụp): lợi thế của cfpair_reg
cũng nằm ở phần "kiểm" (token khác giá trị), không ở giọng văn.
Còn thiếu: (1) đối chứng ORM học từ nhãn đúng/sai trên rollout của chính model cùng số mẫu (y* có thêm gì?); (2) định vị bước sai (cho thưởng dày);
(3) kiểm lách điểm khi policy tối ưu theo bộ kiểm.

## R05-B tiếp: đối chứng ORM, định vị bước, lách điểm (`diag_verifier2`, 27–28/11; `experiments/rounds/R05/verifier2_results.json`)
Cùng 256 câu thử (lời giải mới của model); mỗi bộ kiểm có train dùng 2.738 mẫu/lớp.
| bộ kiểm | AUC trong nhóm | AUC trong nhóm (viết xong) | AUC gộp (viết xong) | **định vị lỗi** (tiền tố sạch vs lỗi) | lách: chỉ đáp án | lách: bỏ giữa | lách: câu tự khẳng định |
|---|---|---|---|---|---|---|---|
| zero-shot | 0,661 | 0,593 | 0,655 | 0,543 | 0,03 | 0,21 | 0,66 |
| **từ y*/ỹ*** | 0,704 | 0,678 | 0,788 | **0,895** | **0,28** | **0,23** | 0,89 |
| ORM (nhãn của chính model) | **0,889** | **0,813** | **0,877** | 0,565 | 0,44 | 0,45 | 0,77 |
| gộp | 0,802 | 0,794 | 0,871 | 0,819 | 0,62 | 0,35 | 0,68 |
(lách = tỉ lệ bản sửa của lời giải SAI được chấm cao hơn bản gốc; > 0,5 = lách được.)
Đọc: ORM thắng ở chấm CẢ BÀI; chỉ bộ kiểm từ y* ĐỊNH VỊ được bước sai (0,895 vs 0,565) → y* cho giám sát theo bước mà nhãn
đúng/sai không cho. ORM và bản gộp dễ bị lách hơn (chỉ ghi đáp án / bỏ giữa); mọi bộ kiểm bị câu tự khẳng định lách → chỉ chấm bước có công thức.
Thiết kế hợp lý: phần thưởng theo bước từ bộ kiểm y* (tách riêng, không gộp) + phần thưởng cuối bài gốc.

## Khoảng cách thầy–trò: cùng y* Luna, model học 0,5B / 1,5B / 3B (28/11; `.work/vm/out_gap/`, 3B lấy từ R02 + `diag_verifier2`)
| model học | cos(SFT y*, GRPO) | cos(SFT lời giải ĐÚNG của chính nó, GRPO) | mốc GRPO–GRPO | bộ kiểm y*/ỹ*: định vị lỗi (zero-shot → học) | AUC đúng/sai trong nhóm, viết xong (zs → học) |
|---|---|---|---|---|---|
| 0,5B | +0,075 | +0,076 | +0,085 | 0,50 → 0,56 | 0,49 → 0,44 |
| 1,5B | **+0,208** | +0,005 | +0,010 | 0,49 → **0,99** (ít mẫu) | 0,50 → 0,77 |
| 3B | **−0,149** | +0,074 | +0,079 | 0,54 → **0,895** | 0,59 → 0,68 |
Cos của cả y* trộn hai thứ: học THỂ THỨC/ĐÁP ÁN (model nhỏ nào cũng cần) và học NỘI DUNG TÍNH TOÁN của thầy. Phải tách theo loại token
mới đọc được (cùng phép đo, `ys_*_cos`; bản đầu đọc thẳng cột "cả y*" và kết luận vội là dự đoán sai — người dùng chỉ ra, đã sửa 28/11):
| model | chữ thường | số trong công thức (giá trị tính ra) | ký hiệu | vùng đáp án | cả y* |
|---|---|---|---|---|---|
| 0,5B | −0,029 | +0,052 | +0,067 | +0,079 | +0,075 |
| 1,5B | +0,184 | **−0,033** | +0,027 | **+0,218** | +0,208 |
| 3B | −0,131 | **−0,055** | −0,122 | +0,001 | −0,149 |
Đọc:
1. Ở 1,5B phần dương đến từ vùng đáp án và chữ (thể thức, trình bày); bắt chước GIÁ TRỊ TÍNH RA của Luna đã hơi ngược chiều GRPO (−0,033);
   ở 3B ngược hơn (−0,055), và cả phần trình bày cũng hại (model đã có giọng riêng). 0,5B còn học cái cơ bản nên mọi phần dương nhẹ.
2. Phần thể thức/đáp án của thầy mang sang được ở model nhỏ; phần NỘI DUNG tính toán thì chuyển sang ngược chiều khi trò bắt đầu có cách
   giải riêng → khớp nguyên lý "mang sang cơ chế, không mang sang chính sách"; nguyên lý KHÔNG bị bác.
3. Mang sang cơ chế (bộ kiểm) cần đủ năng lực: hỏng ở 0,5B, tốt từ 1,5B.
4. 3B (cấu hình capstone): mọi phần trừ đáp án đều ngược chiều → nên mang sang bộ kiểm, không bắt chước cách giải.
Giới hạn: chỉ đo hướng cập nhật ở điểm xuất phát; định vị lỗi ở 1,5B ít mẫu; mỗi cỡ đo một lần.

## Phân tích lời giải từng câu: cfpair_reg vs grpo cùng seed, và vs placebo (28/11; generations @512/@2048, 1.517 câu × 4)
| run | @512: đúng | @512: viết xong | @512: đúng khi viết xong | @2048: đúng | @2048: đúng trên câu cả 8 mẫu viết xong |
|---|---|---|---|---|---|
| grpo s2 | 0,233 | 0,329 | 0,641 | 0,390 | 0,436 |
| cfpair_reg s2 | 0,277 | **0,438** | 0,592 | 0,385 | 0,430 |
| cfpair_reg s1 | 0,274 | 0,414 | 0,614 | 0,390 | 0,430 (vs plc) |
| placebo s1 | 0,263 | 0,412 | 0,587 | 0,368 | **0,403** |
(tỉ lệ gộp theo mẫu, không phải TB 5 bộ.)
Đọc:
1. Lợi thế @512 của cfpair_reg = **hoàn toàn do viết xong**: nếu grpo có tỉ lệ viết xong của reg mà giữ độ đúng-khi-viết-xong của nó →
   0,281 ≥ reg 0,277. Ở 2048 (cả hai viết xong ~97%) độ đúng NGANG NHAU (0,430 vs 0,436). → cfpair_reg KHÔNG làm model suy luận đúng hơn;
   nó làm model viết gọn hơn (~18% ký tự: ít chữ −20%, ít in đậm −27%, ít "verify/check" −29%, ít dòng tính −11%, cùng cấu trúc) mà KHÔNG mất độ đúng.
2. Placebo viết xong y như reg (0,412 vs 0,414) nhưng **mất độ đúng** (0,403 vs 0,430 ở câu viết xong hết, @2048). → tương phản kiểu bất kỳ
   trên y* đều kéo viết gọn; cặp ĐỔI GIÁ TRỊ là thứ giữ được độ đúng trong lúc nén. Vai trò của tính hợp lệ: bảo toàn, không nâng.
3. Trên 1.517 câu @512: reg hơn ở 275 câu, grpo hơn ở 124; ở câu reg hơn, 65% mẫu của grpo bị cắt.
→ Sửa câu chuyện: không phải "học giá trị đúng → giải đúng hơn" mà "nén lời giải về phía văn gọn của thầy mà không làm hỏng tính đúng,
   nhờ cặp phản thực tế chỉ khác tính hợp lệ". Khớp "What Outcome RL Buys" (82% lợi ích là finishing).
**Câu chỉ một bên giải được (@2048, ≥1/4 đúng):** chỉ reg 91 câu, chỉ grpo 108 câu — đối xứng. Bên thua hỏng chủ yếu vì SAI ĐÁP ÁN khi đã
viết xong (92% / 96%), không vì bị cắt. Đọc 8 ví dụ: hai bên cùng hướng giải, khác ở MỘT chỗ trượt (một phép tính, áp sai một tính chất);
bên thắng cũng chỉ đúng 1–2/4 → nhiễu lấy mẫu ở ngưỡng, không thấy kỹ năng riêng của thầy.
**Tỉ lệ phép tính số học sai** (bản chặt, `.work/analysis/arith_check.py`: chỉ trong công thức, tách chuỗi theo "=", chỉ kiểm khi CẢ HAI
vế là biểu thức số trọn vẹn, bỏ số mũ/≈/mod; bản đầu bằng regex bắt nhầm nhiều, ra ~21% — bỏ). Đọc 15 trường hợp bị đánh sai: phần lớn là
trượt thật (138 − 88 = 100; 65537 = 53·1223; 64 = 2·2·2·2·2), vài trường hợp do làm tròn (65/19 = 3,4211) → con số hơi cao hơn thật.
@2048: grpo_s2 **5,8%** phép tính sai, cfpair_reg_s2 **6,6%**, cfpair_reg_s1 8,2%, placebo_s1 7,9%; lời giải sai (viết xong) có ≥1 phép tính
sai kiểm được: 4,8–5,5% ở cả bốn. → (1) cfpair_reg KHÔNG tính đúng hơn khi tự sinh (cùng seed còn hơi nhiều hơn); (2) chỉ ~5% lời giải sai có
lỗi số học kiểm được → lỗi chính là dựng bài/suy luận, không phải tính toán. Ghép với `diag_verifier`: kiến thức "phép tính nào hợp lệ" học
được từ cặp y*/ỹ* (bộ kiểm định vị lỗi 0,895) nhưng KHÔNG chảy vào cách model sinh lời giải qua số hạng DPO.
