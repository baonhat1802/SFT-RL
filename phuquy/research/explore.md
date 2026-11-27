# Explore — đào ý tưởng nhanh, có quy tắc thất bại (capstone)
Bắt chước `Peer_reconstruction/research/explore.md`. Explore nhanh, rẻ, nhưng **mỗi run vẫn có câu hỏi và điểm dừng**, nếu không thành
leo bench mò. Số explore **không** so với bảng nhóm hay bài báo; ý tưởng sống sót mới lên protocol đầy đủ (skill `cs-leaderboard`).

## Profile explore (CHƯA đo — điền sau run GRPO-only đầu tiên trên VM)
| | Giá trị | Lý do |
|---|---|---|
| Khởi đầu | Qwen2.5-3B base | |
| Dữ liệu | `curated` (8.519 câu), cùng thứ tự theo seed | |
| LoRA | r32 α64, mọi lớp tuyến tính | như bảng nhóm |
| GRPO | lr 1e-5, 8 lời giải/câu, phần thưởng `reward.py`, trần 512 token, … bước (điền) | như bảng nhóm, rút số bước cho rẻ |
| Đánh giá | **dev** = MATH500 + Minerva + OlympiadBench (bộ BRIDGE), n = 4, 512 token | AIME24/AMC23 (30/40 câu) chỉ ở protocol đầy đủ: 1 câu AIME = 3,3 điểm, nhiễu hơn mọi cải tiến cần thấy; chọn cấu hình trên bộ báo cáo làm số báo cáo bị thổi phồng |
| Thời gian / chi phí | … giờ/run, … USD (điền) | |

## Mức nhiễu và ngưỡng (CHƯA đo)
Cần: GRPO-only explore × 3 seed → độ lệch chuẩn của TB dev giữa seed (σ_seed), và của hiệu cặp theo seed. Ngưỡng 1 seed = 2 × σ(hiệu);
3 seed = 2 × σ(hiệu) / √3. **Tạm dùng ≥ 2,0 điểm TB dev** cho tới khi đo (nhiễu chỉ riêng việc lấy mẫu khi đánh giá, n = 4, đã là ~0,8–1,5
trên 5 bộ). Thẻ nào muốn ngưỡng khác phải ghi lý do trước khi chạy.

## Quy tắc thất bại
| Tình huống | Nghĩa | Làm gì |
|---|---|---|
| Số đo cơ chế không đổi | cài sai hoặc ý tưởng không làm được điều nó định làm | kiểm code một lần; vẫn không đổi → bỏ |
| Cơ chế đổi, TB không hơn `novel=0` | cơ chế không phải chỗ nghẽn → sai về nguyên lý | ghi kết luận âm tính, chuyển hướng |
| Hơn `novel=0` nhưng dưới ngưỡng | chưa thấy tín hiệu | hết ngân sách thì dừng; **không** thêm seed để "vớt" |
| `novel=0` không ra GRPO-only ± nhiễu | khung cài sai — mọi Δ của vòng vô nghĩa | dừng vòng, sửa khung |

Hai giới hạn (ghi vào thẻ khi áp dụng):
- Explore thành công ≠ bằng chứng → phải lên protocol đầy đủ (3 seed, 5 bộ, 512 và 2.048 token).
- Explore thất bại có thể là **âm tính giả nếu ý tưởng cần train lâu** (explore rút bước) → ghi "phụ thuộc số bước: có/không".
