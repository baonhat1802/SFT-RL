# Schölkopf et al., "Toward Causal Representation Learning" (Proc. IEEE 2021, arXiv 2102.11107) — đọc toàn văn 27/11

## Các nguyên lý chính (theo bài)
- **Thang mô hình** (Bảng I): thống kê (dự đoán i.i.d.) < đồ thị nhân quả (dự đoán dưới can thiệp) < SCM (trả lời phản thực tế) < cơ học.
- **SCM** X_i := f_i(PA_i, U_i), nhiễu độc lập → **phân rã nhân quả** P(X_1..X_n) = Π P(X_i | PA_i): mỗi thừa số là một **cơ chế**.
  Can thiệp = thay một phép gán (cứng: hằng số; mềm: đổi hàm/nhiễu). Phản thực tế cần cố định nhiễu U.
- **Nguyên lý cơ chế nhân quả độc lập (ICM)**: các cơ chế là module tự trị, không thông tin/ảnh hưởng nhau — đổi (can thiệp) một cơ chế không
  đổi cơ chế khác; biết cơ chế này không cho biết cơ chế kia.
- **Giả thuyết dịch chuyển cơ chế thưa (SMS)**: thay đổi phân phối nhỏ thường chỉ đổi MỘT VÀI thừa số của phân rã nhân quả; trong phân rã không
  nhân quả ("vướng"), mọi thừa số cùng đổi. Locatello 2020: SMS đủ để nhận dạng biến ẩn khi có dữ liệu phù hợp.
- **Học biến nhân quả**: X = G(S_1..S_n); biến nào nhận dạng được tuỳ vào can thiệp/dịch chuyển quan sát được.
- **Học cơ chế chuyển giao được**: kiến trúc module phản chiếu tính module của thế giới; cơ chế dùng lại qua nhiệm vụ.
- **Hệ quả**: SSL chỉ có ích theo hướng ngược nhân quả (ICM: P(X) không chứa thông tin về P(Y|X) theo hướng nhân quả); bền vững = rủi ro xấu nhất
  trên tập phân phối can thiệp (pt. 17–18), tăng cường dữ liệu = can thiệp nhân tạo; RL on-policy ước lượng xác suất do(·); RL nên dùng phản thực
  tế ("hành động trong không gian tưởng tượng") và mô hình thế giới; offline RL là bài toán phản thực tế.

## Đối chiếu với những gì capstone đã làm (thẳng thắn)
Đã dùng: công cụ CẤP CAN THIỆP/PHẢN THỰC TẾ và ước lượng — cặp ghép phản thực tế (R03-1), đối chứng âm/placebo, AIPW (R04-1), CATE (R04-3),
đồng nhất thức Bayes φ = log Q/V. CHƯA dùng: ICM, SMS, phân rã nhân quả của quá trình giải toán thành các cơ chế, và câu hỏi "can thiệp lên
cơ chế NÀO" — tức phần cốt lõi của bài.

## Ánh xạ sang SFT + RL cho suy luận
Quá trình giải = chuỗi cơ chế: chọn hướng đi (kế hoạch / nước đi đầu) → thực hiện từng bước → viết đáp án. SFT trên y* thay TOÀN BỘ phân phối
của trò bằng của Luna (một dịch chuyển dày: giọng, ký hiệu, thực hiện, kế hoạch cùng đổi). SMS gợi ý: cái cần đổi để giải câu khó là THƯA.
Bằng chứng sẵn có: một câu đầu của y* (+0,065, cứu 41% vs 25% câu cả-sai) trong khi thầy thấy cả y* chỉ đổi ~1% token và lời nhắc chung không giúp
→ dịch chuyển cần thiết nằm ở cơ chế "chọn hướng đi".
