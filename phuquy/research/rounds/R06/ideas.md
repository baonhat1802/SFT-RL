# R06 — ý tưởng sau phép kiểm "gộp dữ liệu" (2026-11-28)

Khung chung (người dùng duyệt 28/11): rollout của trò = dữ liệu THỰC NGHIỆM (không lệch, rỗng ở nhóm cả-sai); lời giải thầy = dữ liệu QUAN SÁT
có biến gây nhiễu ẩn (hiểu biết riêng của thầy). Bài toán = dùng dữ liệu quan sát ở nơi dữ liệu thực nghiệm rỗng.

Bằng chứng dùng (đều trong `research/STATE.md` 28/11):
- E1 `diag_fusion` T2 trượt: phần lệch học ở câu lưng chừng KHÔNG chuyển sang câu cả-sai; phần prior theo loại token (hệ số chặn) làm hỏng.
- E2 tín hiệu thầy THÔ trên 8 bài sai cùng hướng GRPO-32 (cos 0,150; mốc nhiễu 0,026) → tín hiệu thầy có chuyển được.
- E3 độ dốc theo tín hiệu thầy ở token chữ ≈ 0, ở token toán/số/đáp án 0,04–0,08.
- E4 D4 (hai lần): tương quan quan sát phóng đại hiệu ứng can thiệp ~3 lần (biến ẩn "trò đang đi đúng hướng").
- E5 42% câu cả-sai ở 8 mẫu có bài đúng khi thêm 24 mẫu.
Đang chạy (VM d, xong tự tắt): `diag_fusion --diag 1`, tách phép hiệu chỉnh thành độ dốc / hệ số chặn / chỉ token toán, lưu dữ liệu từng mẫu.

## R06-1: Chuyển tín hiệu thầy bằng TÍNH BẤT BIẾN qua độ khó — method `r06_invtransport`
Vấn đề: E1 — mối liên hệ "tín hiệu thầy → advantage" học ở câu dễ không đúng ở câu khó.
Nguyên lý: cái gì nhân quả thì bất biến qua môi trường (Peters 2016 ICP; Arjovsky 2019 IRM). Rojas-Carulla et al. 2018: khi miền đích KHÔNG có
nhãn, dự đoán chỉ từ tập đặc trưng bất biến là tối ưu trong trường hợp xấu nhất. Ở đây: miền đích = nhóm cả-sai (không có advantage);
môi trường = tầng độ khó (1–2/8, 3–5/8, 6–7/8 đúng; thêm tầng "cả-sai nhưng cứu được ở 32 mẫu" tạo rẻ bằng mẫu thêm).
Cơ chế: tìm các đặc trưng của tín hiệu thầy (theo loại token, vị trí, …) có quan hệ với advantage KHÔNG ĐỔI qua các tầng (kiểm ICP hoặc phạt
V-REx); chỉ dùng phần đó làm phần thưởng theo token ở nhóm cả-sai trong GRPO. Không có gì riêng cho toán: "đặc trưng" là bất cứ gì đo được.
Khác `r03_invar` (đã chết ở kiểm trước): đó là bất biến của GRADIENT qua GIỌNG VĂN (y* vs lời giải tự sinh); đây là bất biến của QUAN HỆ tín hiệu
thầy → advantage qua ĐỘ KHÓ, để chuyển sang miền không nhãn.
Gần nhất: Invariant Gradient Alignment 2606.05025 (môi trường = nhóm bài đồng cấu tự tạo, chưng cất, không RL); Invariance-Weighted Distillation
2609.22566 (trọng số mẫu theo độ ổn định của thầy qua nhiễu loạn tổng hợp). Chưa thấy bài dùng tầng độ khó làm môi trường để chuyển tín hiệu
thầy vào nhóm cả-sai của RL.
Kiểm trước (dữ liệu MỚI, seed khác, tiêu chí viết trước): tín hiệu chọn bằng bất biến phải hơn tín hiệu thô ≥ 0,05 cos ở T2.

## R06-2: Dùng thầy có BẢO HIỂM trước nhiễu (phân tích độ nhạy) — method `r06_robust`
Vấn đề: E1 + E4 — không biết thầy lệch bao nhiêu ở nhóm cả-sai.
Nguyên lý: mô hình độ nhạy biên (marginal sensitivity model, Tan 2006; Kallus & Zhou 2018–2020): giả định hiểu biết ẩn của thầy làm lệch tỉ lệ
xác suất tối đa Γ lần → tín hiệu thầy thật nằm trong khoảng [d − log Γ, d + log Γ]. Chính sách học theo cận an toàn (bi quan) của khoảng đó.
Cơ chế: phần thưởng theo token ở nhóm cả-sai = phần của tín hiệu thầy vượt khỏi biên nhiễu (dấu chắc chắn với mọi độ lệch ≤ Γ); Γ ƯỚC LƯỢNG
từ nhóm lưng chừng, nơi đo được độ lệch thật (Γ nhỏ nhất để cận an toàn không bao giờ trái dấu với RL). Γ → ∞ = GRPO; Γ = 1 = tự chưng cất thô.
Gần nhất: robust fitted-Q dưới mô hình độ nhạy (2302.00662, gợi ý dùng cận làm khởi động cho RL online); "An Empirically Sharper Estimator for
Confounding-Robust RL" (NeurIPS 2026). Không thấy bài cho LLM / SFT+RL.
Kiểm trước: như R06-1, so cận an toàn với tín hiệu thô ở T2; neo: Γ = 1 phải ra đúng số của tín hiệu thô.

## R06-3: Gán công cho từng bước bằng suy luận nhân quả qua BIẾN ĐẠI DIỆN (proximal) — method `r06_proximal`
Vấn đề: E4 — bước "trông quan trọng" phần lớn vì biến ẩn "trò đang đúng hướng".
Nguyên lý: suy luận nhân quả qua biến đại diện (Miao, Geng, Tchetgen Tchetgen 2018): có hai biến đại diện nhiễu của biến ẩn thì nhận diện được
hiệu ứng nhân quả mà không cần can thiệp. Ở đây có sẵn hai biến đại diện: độ tự tin của trò ở đoạn trước và xác suất thầy gán cho đoạn trước.
Cơ chế: ước lượng hiệu ứng khử nhiễu của từng đoạn lên việc giải đúng (hàm cầu nối), dùng làm công theo đoạn trong GRPO.
Gần nhất: MBRL dưới nhiễu dùng proximal OPE (2512.07528), khảo sát causal RL (2512.18135) — không có cho LLM. Rủi ro: nặng giả thiết, phức tạp.

## R06-4: GRPO phản thực — thầy là HÀNH ĐỘNG THAY THẾ trong cặp song sinh cùng nhiễu — method `r06_cfgrpo`
Người dùng (28/11): "phải can thiệp vào GRPO như vậy mới đúng" — giữ phản thực làm lõi, học qua advantage của GRPO, không qua DPO.
Khung: gộp dữ liệu phản thực (Forney, Pearl, Bareinboim 2017): quan sát (y*), thực nghiệm (rollout), PHẢN THỰC (song sinh cùng nhiễu).
Cơ chế: nhóm cả-sai (GRPO = 0 gradient). Với mỗi bài sai τ (nhiễu U đã biết, SCM Gumbel-max theo vị trí token của trò), tại điểm cắt t:
  - hành động thật: đoạn trò viết tiếp τ[t:] (Y = 0);
  - hành động phản thực: chèn bước của thầy a' NHƯ CHỮ CỦA TRÒ (không phải ghi chú), trò viết tiếp c' với CÙNG nhiễu → Y_cf.
  Bước chèn = bước sớm nhất của y* mà các số của nó chưa xuất hiện trong τ[:t] ("bước kế tiếp trò còn thiếu"), bỏ bước chứa đáp án.
  Nhóm GRPO ở trạng thái s_t = {τ[t:], a'+c'}; advantage = kết quả − trung bình nhóm. c' là on-policy (sinh bởi trò với tiền tố s_t+a');
  chỉ token a' là off-policy (cắt tỉ lệ). Tiền tố chung τ[:t] không nhận gradient → công dồn đúng vào chỗ rẽ.
Bằng chứng: D3 (bản giữ số): chèn sự kiện ở 25/50/75% bài sai lật 30/22/27% vs placebo 1/1/4%, cặp chung 25/50/76%; chèn rỗng tái tạo 100%.
Gần nhất: TWICE 2610.31544 (ĐÃ RÚT, chỉ có tóm tắt): song sinh nhiễu Gumbel chung để gán công theo lượt cho agent QA (Qwen2.5-3B/7B);
  hành động thay thế là quyết định KHÁC CỦA CHÍNH policy, không có thầy / lời giải mẫu, không nói nhóm cả-sai → khác: hành động thay thế = bước
  của thầy (đặc quyền), mục tiêu = nhóm cả-sai, gộp SFT vào RL. BREAD 2506.17211: tiền tố thầy ở ĐẦU bài (nửa đầu y*, tìm độ dài bằng chia đôi),
  không ghép nhiễu, không chèn giữa bài sai của trò. Relay-OPD / MENTOR 2610.13594 / Find-Fix-Reason 2604.16243 / CausalOPD 2608.03673: thầy can
  thiệp bài sai nhưng cần model thầy, không có advantage phản thực cùng nhiễu. CCA-PG 2011.09464: baseline có điều kiện tương lai + số ngẫu
  nhiên chung, không LLM, không thầy.
Kiểm trước (đề xuất, dữ liệu mới): câu cả-sai ở 8 mẫu nhưng giải được ở 32 mẫu; gradient GRPO-phản-thực tính trên 8 bài sai + song sinh,
  so cos với GRPO-32; đối chứng: hành động placebo (bước của câu khác), tiền tố thầy ở đầu bài không ghép nhiễu (kiểu BREAD), tín hiệu thầy thô.

## R06-5: GRPO với trọng số OVERLAP (lõi bộ ước lượng) — method `r06_overlap`
Người dùng (28/11): R06-4 "vẫn ở bọc ngoài, chỉ xử lý rollout, chưa vào lõi" → sửa chính bộ ước lượng.
Chẩn đoán nhân quả: A của GRPO = ước lượng hiệu ứng hành động; chỉ có dữ liệu từ chính trò → hành động trò không bao giờ chọn KHÔNG nhận diện
được (vi phạm overlap/positivity) = cách phát biểu nhân quả của "RL bị chặn bởi model gốc" (Yue 2025); SFT = ngoại suy bằng dữ liệu nhiễu.
Cơ chế: tại trạng thái s của bài sai, hai hành động: dòng tiếp của trò a (π, q) và bước kế tiếp còn thiếu của thầy a' (π, q); q = cùng model
thấy y*. Kết quả phản thực cùng nhiễu. Trọng số overlap (Li–Morgan–Zaslavsky 2018) cho "thầy đề xuất" vs "trò đề xuất": w(a') = π/(π+q),
w(a) = q/(π+q); Ȳ = trung bình có trọng số; gradient Σ w·(Y − Ȳ)·∇log π. SFT và RL thành MỘT bộ ước lượng của hiệu ứng trên vùng chồng lấn;
q = π → GRPO; hành động ngoài tầm trò → trọng số ≈ 0 (không ép như SFT); vùng chồng lấn nở khi trò giỏi lên.
Kiểm: `code/beat/diag_overlap.py` (seed 72, bộ câu mới) — P1 vùng chồng lấn không rỗng (≥ 20% bước thầy có w ≥ 0,05, lật ≥ 2× placebo);
P2 cos(overlap, GRPO-32) > không trọng số + 0,03, > placebo + 0,05, > 0. Đối chứng thêm: trọng số importance π/q (không lệch, dự đoán ≈ 0).
Rủi ro đã biết trước: bước thầy nhiều token → log q − log π cộng dồn lớn → w ≈ 0 gần hết (P1 trượt).

## R06-6: Thầy là BIẾN CÔNG CỤ, hành động định nghĩa theo NỘI DUNG — method `r06_iv`
Sửa R06-5 (bản thu nhỏ: trọng số overlap 1e-17…1e-12 vì hành động = CHUỖI CHỮ của Luna; model thấy y* chép nguyên văn → overlap rỗng do
định nghĩa treatment quá mịn, không phải bản chất). Treatment F = "đoạn tiếp theo xác lập sự kiện kế tiếp còn thiếu của thầy" (theo số, viết
kiểu gì cũng được). Biến công cụ Z bốc thăm: đoạn đó viết với đề thường (Z=0) / với y* (Z=1) / với y* của câu khác (placebo); sau đoạn, ngữ
cảnh trở lại thường. LATE = RF/FS = hiệu ứng của F ở trạng thái mà gợi ý làm trò đổi cách làm (vùng với tới được); khử biến ẩn "đang đúng
hướng" (D4). Vào GRPO: công của đoạn theo LATE thay vì chia đều phần thưởng cả bài.
Kiểm: `code/beat/diag_iv.py` (seed 73): I1 FS ≥ 0,10, CI > 0, placebo FS ≈ 0; I2 ≥ 20% trạng thái có P(F|Z=0) > 0 và P(F|Z=1) > 0;
I3 LATE CI > 0 và |LATE − tương quan thô| ≥ 0,10. Kiểm loại trừ: tỉ lệ đoạn chứa đáp án theo từng nhánh (Z=1 lộ đáp án → vi phạm loại trừ).
