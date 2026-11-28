# Vòng R04 — brainstorm nhân quả mở rộng (2026-11-27, chưa kiểm trước)
Trọng tâm (người dùng): "hiểu cơ chế nhân quả để moi kiến thức cũ ra dùng tốt hơn, học thêm cơ chế nhân quả từ dense feedback, tự đặt
counterfactual" — không bó vào nội dung/giọng văn, không bó vào độ dài.

| # | Góc nhân quả | Cơ chế một câu | Vấn đề đo được | Gần nhất (rà sơ bộ) | Kiểm trước |
|---|---|---|---|---|---|
| R04-1 | Ước lượng doubly robust (AIPW) | thầy-có-y* = mô hình kết quả, phần thưởng = nhãn vàng; log-ratio thầy làm control variate phụ thuộc hành động, bù chính xác trên cả từ vựng → kỳ vọng cập nhật = GRPO đúng, thầy chỉ giảm nhiễu, thiên lệch của thầy (placebo, giọng) không lọt vào | nhiễu GRPO: cos 0,04–0,29; thầy bị ngữ cảnh chi phối (placebo) | vOPD (baseline log-ratio KHÔNG phụ thuộc hành động, cho OPD), Stein CV 1710.11198, "Mirage" 1802.10031 (cảnh báo lợi ích nhỏ) | diag: cos(g_DR(A), g_DR(B)) vs GRPO-GRPO |
| R04-2 | Phân tích trung gian (mediation) của tác động y* | y* trong ngữ cảnh làm tỉ lệ đúng 28 → 35%; tìm quyết định nào của chính model-có-y* mang lợi ích đó (ép trò lấy quyết định ấy rồi tự viết tiếp) → phần thưởng dày = hiệu ứng gián tiếp, chỗ rẽ = trung gian mạnh nhất | allfail ~30%; diag_fork: rẽ trên lời giải SAI vô ích | InT 2601.14209, KnowRL 2604.12627, QuestA | diag: cứu câu cả-sai khi ghép tiền tố model-có-y* tại trung gian vs cùng độ dài ngẫu nhiên vs tiền tố y* |
| R04-3 | Hiệu ứng can thiệp không đồng nhất (CATE) để moi kiến thức cũ | model tự can thiệp lên ngữ cảnh (nhắc lại đề, liệt kê định lý, giải trường hợp riêng, đi ngược từ dạng đáp án, nước đi đầu của y*); chia nhóm GRPO thành thí nghiệm ngẫu nhiên giữa các can thiệp; học chính sách chọn can thiệp theo CATE, rồi để model tự viết can thiệp đó ở bước đầu | 24% câu cả-sai được cứu chỉ bằng thêm mẫu → kiến thức có sẵn, thiếu "gợi nhớ" | RLAD (abstraction), gợi ý chiến lược; chọn theo CATE chưa thấy | sinh: 256 câu khó × 5 can thiệp × 8 mẫu; ATE và độ dự đoán được của CATE |
| R04-4 | Tính cần thiết (probability of necessity) của từng bước | tự phá từng bước của lời giải đúng (do(bước := hỏng)), viết tiếp; bước nào hỏng mà vẫn đúng là không cần → công lao dồn vào bước cần thiết | GRPO thưởng đều mọi token | 2601.07408, 2607.27888 (credit phản thực tế) — gần, cần đọc | — |
| R04-5 | Huấn luyện can thiệp hoán đổi (causal abstraction, IIT) với giá trị trung gian của y* | ép trạng thái ẩn tại một giá trị trung gian quyết định nhân quả phần viết sau | — | Geiger 2112.00826; cần nhãn phản thực tế nhất quán → nặng | — |

## Kiểm trước R04-1 (27/11, `runs/diag_dr`, dừng sau 2/3 batch — kết luận đã rõ) — LOẠI
192 mẫu mỗi nửa (cả mẫu advantage 0). Mốc cos(GRPO_A, GRPO_B) +0,062 / +0,087.
| | batch 1 (đầy đủ / ngắn gọn) | batch 2 |
|---|---|---|
| neo: cos(CV_A, CV_B) ≈ 0 | +0,02 / −0,00 | −0,03 / −0,04 |
| neo: cos(CV_A, GRPO_B) ≈ 0 | +0,04 / +0,01 | −0,04 / +0,02 |
| cos(CV, GRPO) cùng mẫu (điều kiện giảm nhiễu) | −0,05 / −0,06 | **+0,05 / +0,06** (đổi dấu) |
| β* → cos(DR_A, DR_B) | +0,068 / +0,071 | +0,089 / +0,084 |
Bộ máy đúng (hai neo ≈ 0: thầy không làm lệch cập nhật), nhưng tương quan giữa control variate và nhiễu GRPO chỉ |ρ| ≈ 0,05 và ĐỔI DẤU giữa batch
→ giảm phương sai tối đa ρ² < 0,5%. Thầy-có-y* của model base không biết thêm gì ở đúng chỗ quyết định thành bại (khớp: phản ứng ~1% token,
placebo ≈ thầy thật). Log `experiments/rounds/R04/diag_dr.log`.

## Kiểm trước R04-3 (27/11, `experiments/rounds/R04/cate_results.json`) — can thiệp TỰ ĐẶT: LOẠI; câu đầu của y*: tác động lớn
384 câu → 224 câu khó (≤ 2/8 đúng; 122 câu cả-sai). Mỗi nhánh 8 mẫu @2048.
| nhánh | đúng TB | cứu câu cả-sai | Δ so với lấy mẫu thường (KTC 90%) | độ dài |
|---|---|---|---|---|
| lấy mẫu thường (8 mẫu mới) | 0,143 | 0,254 | — | 662 |
| nhắc lại đề | 0,133 | 0,311 | −0,009 [−0,027; +0,007] | 678 |
| liệt kê định lý | 0,122 | 0,246 | −0,021 [−0,039; −0,003] | 757 |
| trường hợp riêng | 0,123 | 0,246 | −0,020 [−0,038; −0,002] | 709 |
| đi ngược từ dạng đáp án | 0,132 | 0,303 | −0,011 [−0,026; +0,006] | 660 |
| **câu đầu của y\*** (gợi ý) | **0,207** | **0,410** | **+0,065 [+0,045; +0,085]** | 693 |
Giá trị chính sách (chọn trên 4 mẫu, chấm trên 4 mẫu khác): thường 0,136; nhánh tốt nhất chung (câu đầu y*) 0,213; chọn theo môn 0,213 (môn nào
cũng chọn câu đầu y*); chọn theo từng câu 0,161 (KÉM chọn chung) → không có hiệu ứng không đồng nhất khai thác được.
Đọc: model base KHÔNG tự moi được kiến thức bằng các lời nhắc chung (còn hơi hại, dài hơn). Nhưng MỘT câu của y* (nước đi đầu: đặt ẩn, chọn công thức,
chia trường hợp) tăng tỉ lệ đúng +45% tương đối và cứu 41% câu cả-sai (vs 25%) — trong khi thầy-có-CẢ-y* chỉ đổi ~1% token. Nút thắt ở câu khó là
CHỌN NƯỚC ĐI ĐẦU, và y* chứa đúng thông tin đó.

## Chẩn đoán vì sao R04-1 hỏng (`diag_phi`, 27/11, 256 câu × 8 mẫu, 171 nhóm lẫn; `experiments/rounds/R04/phi_results.json`)
| | thầy đầy đủ | thầy ngắn gọn | placebo (y* câu khác) |
|---|---|---|---|
| AUC trong nhóm, TB φ cả lời giải | 0,657 | 0,669 | **0,643** |
| AUC, phần trước `\boxed` | 0,640 | 0,647 | **0,648** |
| AUC, chỉ vùng `\boxed` | 0,602 | 0,628 | **0,475** |
| tương quan φ (đã căn giữa) với A theo TỪNG TOKEN | 0,007 | 0,006 | 0,004 |
| tương quan tổng φ với A theo CHUỖI | 0,162 | 0,144 | 0,091 |
| token có \|φ\| > 1 | 1,2% | 1,3% | 0,5% |
(AUC của −độ dài: 0,443 — trong cùng nhóm, lời giải ngắn KHÔNG đúng hơn.)
Kết luận: (1) thầy CÓ phân biệt đúng/sai (AUC ~0,66) nhưng thầy placebo cũng làm được gần y hệt (0,64) — phần phân biệt đó đến từ KHUNG PROMPT
(token lớn nhất: chữ rác, đổi ngôn ngữ, lặp — thầy nào cũng phạt văn bản thoái hoá), không từ nội dung y*. (2) Phần riêng của y* chỉ nằm ở vùng
đáp án (0,60–0,63 vs placebo 0,47) = thầy biết đáp án cuối — trùng với thông tin phần thưởng đã có. (3) Ở cấp token tương quan với advantage ≈ 0,006
→ control variate theo token vô dụng; thông tin chỉ có khi cộng dồn cả chuỗi (0,16). Đọc nhân quả: φ bị khung prompt gây nhiễu; thành phần do y*
gây ra là chính KẾT QUẢ (đáp án) — không có tín hiệu cho các bước trung gian.

## Hoán đổi câu mở đầu (`diag_opening`, 27/11, 384 câu → 214 câu khó, 120 câu cả-sai; `experiments/rounds/R04/opening_results.json`)
Câu hỏi: lợi ích của y* có nằm ở CÂU MỞ ĐẦU (cơ chế "chọn nước đi đầu", giả thuyết dịch chuyển cơ chế thưa SMS) không? Mọi nhánh do model
KHÔNG có y* viết tiếp, 8 mẫu mỗi câu.
| nhánh | đúng | Δ so với thường [KTC 90%] | cứu câu cả-sai | độ dài |
|---|---|---|---|---|
| thường (8 mẫu mới) | 0,124 | — | 0,208 | 725 |
| câu đầu Luna làm GỢI Ý | 0,165 | +0,040 [+0,020; +0,060] | 0,292 | 702 |
| câu đầu Luna làm ĐẦU LỜI GIẢI | 0,069 | **−0,055** [−0,075; −0,036] | 0,142 | **374** |
| mở đầu tự viết (không y*) rồi viết tiếp | 0,116 | −0,008 [−0,025; +0,009] | 0,233 | 652 |
| mở đầu tự viết CÓ y* trong ngữ cảnh, viết tiếp KHÔNG y* | 0,150 | +0,025 [+0,006; +0,043] | 0,317 | 688 |
| cả lời giải có y* trong ngữ cảnh | 0,685 | +0,560 | 0,992 | 768 |
Phần qua trung gian câu mở đầu (≈64 token) = **4,5%** tổng hiệu ứng của y* trong ngữ cảnh; câu đầu Luna đặt thẳng vào lời giải: −10%.
Lộ đáp án trong câu mở đầu: 1,1% (cả hai loại) — không phải rò.
Đọc:
1. Giả thuyết "y* tác động qua một cơ chế thưa ở đầu lời giải" **bị bác** ở mức 64 token: hiệu ứng của y* trải dọc lời giải (hoặc nằm ở phần sau).
   Câu mở đầu của model có-y* phần lớn là chép lại đề ("To find …, we follow these steps:") — nước đi thật chưa xuất hiện trong 64 token đầu.
2. Câu của Luna giúp khi là GỢI Ý nhưng HẠI khi bị ép làm đầu lời giải (độ dài giảm một nửa → model chuyển sang giọng cụt của Luna rồi sai) —
   lại thấy giọng văn của y* hại khi đi thẳng vào chuỗi sinh, khớp với chẩn đoán gradient SFT (mục 6 của mentor_updates).
3. Δ gợi ý câu đầu lần này +0,040, nhỏ hơn R04-3 (+0,065) — cùng chiều, khác mẫu câu.
4. Hiệu ứng tổng +0,56 với tỉ lệ cứu 99% chủ yếu là CHÉP đáp án/lời giải; không phải mốc đạt được nếu không có y*.
Hướng kế (chưa chạy): đường cong trung gian theo độ dài tiền tố (16/32/64/128/256 token và 25/50/75% lời giải có-y*) — chỉ ra y* vào ở bước
nào; nếu đường cong tăng đều thì không có cơ chế thưa nào để tách riêng.

## Đường cong trung gian theo độ dài tiền tố (`diag_mediation`, 27/11, 384 câu → 247 câu khó; `experiments/rounds/R04/mediation_results.json`)
Lời giải do model CÓ y* viết, cắt sau k token / f% độ dài, model KHÔNG có y* viết tiếp. Mốc: thường 0,130; có-y*-cả-bài 0,655 (tổng +0,525).
| điểm cắt | token | đúng | Δ | phần trung gian | lộ đáp án | đúng khi bản gốc ĐÚNG | đối chứng: cắt lời giải THƯỜNG đúng |
|---|---|---|---|---|---|---|---|
| 16 tok | 16 | 0,138 | +0,008 | 2% | 0% | 0,17 | — |
| 64 tok | 63 | 0,160 | +0,030 | 6% | 1% | 0,20 | 0,39 |
| 128 tok | 124 | 0,186 | +0,056 | 11% | 3% | 0,23 | — |
| 25% | 185 | 0,180 | +0,050 | 9% | 3% | 0,22 | 0,46 |
| 256 tok | 246 | 0,217 | +0,087 | 17% | 8% | 0,29 | — |
| 50% | 370 | 0,227 | +0,097 | 18% | 8% | 0,30 | 0,57 |
| 75% | 555 | 0,294 | +0,163 | 31% | 14% | **0,40** | **0,69** |
(Cắt lời giải thường rồi viết tiếp: Δ ≈ 0 ở mọi điểm — ép tiền tố tự nó không giúp không hại.)
Đọc:
1. **Không có bước then chốt.** Phần trung gian tăng đều theo độ dài, không có chỗ nhảy. Đến 75% lời giải có-y* mới chuyển được 31% hiệu ứng.
2. **Lời giải có-y* KHÔNG tự đứng được.** Cho model thường 75% một lời giải có-y* ĐÚNG, nó viết tiếp đúng chỉ 40%; với 75% lời giải ĐÚNG của chính nó: 69%.
   Lời giải có-y* chứa các bước "nhảy" chỉ hợp lý khi nhìn thấy y* (đáp án/giá trị lấy từ y*), nên ~70% hiệu ứng tổng nằm ở đoạn cuối = chép kết quả.
   Đọc nhân quả: văn bản có-y* bị y* (biến người học không thấy) chi phối trực tiếp ở từng bước → bắt chước nó là học từ dữ liệu bị gây nhiễu;
   khớp với mục φ (thông tin riêng của y* nằm ở vùng đáp án) và với SFT-trên-y* hại.
3. Phần chuyển được thật (không dựa vào việc thấy y*) ≈ +0,10 ở nửa lời giải — lớn hơn câu mở đầu (+0,03) nhưng tích luỹ dần.
