# Related work — gộp SFT và RL, chưng cất on-policy, tự chưng cất

Tìm ngày 2026-11-11, xuất phát từ BRIDGE (`01`). Mô tả dưới đây lấy từ abstract và trang tóm tắt, **chưa đọc
toàn văn**. Kiểm lại trước khi trích số. PDF không đưa vào git.

## Ba họ phương pháp, nói gọn

- **Gộp SFT + RL một giai đoạn.** Học từ lời giải mẫu có sẵn (do người hoặc model mạnh viết, *off-policy*) cùng lúc
  với RL từ phần thưởng kiểm chứng được (RLVR, ví dụ đáp án đúng/sai). Khó ở chỗ cân hai loss. BRIDGE thuộc họ này.
- **Chưng cất on-policy (OPD).** Trò tự sinh câu trả lời, thầy chấm *từng token* (thường bằng reverse KL). Có cái
  hay của RL (học trên chính phân phối của mình) cộng tín hiệu dày (mỗi token một điểm, thay vì một điểm cho cả câu).
  Cần một model thầy riêng.
- **Tự chưng cất on-policy (OPSD).** Thầy chính là model đó, nhưng được xem thêm *thông tin đặc quyền* trong ngữ cảnh
  (lời giải mẫu, phản hồi môi trường, một lần thử thành công khác trong nhóm). Không cần model thầy. Hướng mới nhất
  là **ghép với GRPO**: phần thưởng quyết định *chiều* cập nhật, thầy chỉ quyết định *độ lớn* theo từng token.

## 1. Gộp SFT + RL một giai đoạn

| Paper | ID | Ý chính |
|---|---|---|
| BRIDGE — Chen et al., ICML 2026 | [2509.06948](https://arxiv.org/abs/2509.06948) | Hai tầng: LoRA "thầy" SFT học để tối đa reward thêm so với chỉ RL. Bản cũ hơn có lẽ là "Learning Useful Supervision for RL in Reasoning Models" (ICLR 2026), chưa xác nhận |
| LUFFY — Yan et al., NeurIPS 2025 | [2504.14945](https://arxiv.org/abs/2504.14945) | Trộn lời giải mẫu vào nhóm GRPO, importance sampling có chuẩn hoá. Baseline gốc của họ này |
| SRFT — 2025 | [2506.19767](https://arxiv.org/abs/2506.19767) | Một giai đoạn, trọng số SFT/RL theo entropy |
| CHORD — 2025 | [2508.11408](https://arxiv.org/abs/2508.11408) | SFT là loss phụ trong vòng RL on-policy, hệ số chung μ + trọng số từng token |
| AMFT — 2025 | [2508.06944](https://arxiv.org/abs/2508.06944) | Học trọng số SFT/RL μ bằng meta-gradient (gần BRIDGE nhất về tinh thần). Có bản bị rút, kiểm phiên bản. Code: `hlxtsyj/AMFT` |
| HPT — 2025 | [2509.04419](https://arxiv.org/abs/2509.04419) | SFT và RL là hai trường hợp của cùng một bài tối ưu, chuyển qua lại theo kết quả rollout |
| ReLIFT, Prefix-RFT, TRAPO | — | Xen kẽ SFT vào RL theo từng câu hỏi khó / theo tiền tố lời giải. Chưa kiểm ID |

## 2. Chưng cất on-policy (thầy là model khác)

| Paper | ID | Ý chính |
|---|---|---|
| On-Policy Distillation — Thinking Machines (blog, 2025) | [link](https://thinkingmachines.ai/blog/on-policy-distillation/) | Bài phổ biến hoá OPD: reverse KL từng token, rẻ hơn RL nhiều |
| GKD — Agarwal et al. 2024 | [2306.13649](https://arxiv.org/abs/2306.13649) | Gốc của chưng cất trên mẫu trò tự sinh (đã có ở individual-project) |
| KDRL — 2025 | — | Gộp KD reverse KL + GRPO trong một mục tiêu. Chưa kiểm ID |
| MOPD — Xiaomi MiMo, ICML 2026 | [2606.30406](https://arxiv.org/abs/2606.30406) | RL riêng từng domain thành nhiều thầy, rồi OPD gộp vào một trò. Thầy cùng gốc mới ổn định |
| TREK — 2026 | [2607.05339](https://arxiv.org/abs/2607.05339) | Câu khó mà trò không giải ra: lấy lời giải của thầy (hoặc chính nó có thêm ngữ cảnh), forward-KL ngắn, rồi về GRPO. Qwen3-8B AIME25 36,9 → 40,3 |
| Demystifying OPD — 2026 | [2607.13399](https://arxiv.org/abs/2607.13399) | Phân tích vai trò và bệnh lý của OPD |
| Weak-to-Strong via Direct OPD — 2026 | [2607.05394](https://arxiv.org/abs/2607.05394) | OPD với thầy yếu hơn trò |

## 3. Tự chưng cất on-policy (thầy = chính model + thông tin đặc quyền)

| Paper | ID | Ý chính |
|---|---|---|
| SDFT — Shenfeld et al. 2026 | [2601.19897](https://arxiv.org/abs/2601.19897) | Thầy có lời giải mẫu trong ngữ cảnh; quên ít hơn SFT (đã có ở individual-project) |
| SDPO — Hübotter et al. 2026 | [2601.20802](https://arxiv.org/abs/2601.20802) | Thầy có phản hồi môi trường (đã có ở individual-project) |
| OPSD / Self-Distilled Reasoner — Zhao et al., ICML 2026 | [2601.18734](https://arxiv.org/abs/2601.18734) | Thầy thấy lời giải tham khảo, trò không. Báo hiệu quả token gấp 4–8 lần GRPO |
| RLSD — 2026 | [2604.03128](https://arxiv.org/abs/2604.03128) | **Reward quyết định chiều, thầy quyết định độ lớn từng token.** Nói OPSD thuần bị rò lời giải và train không ổn định; báo thắng OPSD và SDPO |
| SRPO — 2026 | [2604.02288](https://arxiv.org/abs/2604.02288) | Ghép SDPO + GRPO; tự chưng cất trên mẫu đã đúng gây mơ hồ |
| HSD (Path-Conditioned) — 2026 | [2606.15576](https://arxiv.org/abs/2606.15576) | Thầy xem một lần thử *thành công* trong cùng nhóm, credit dồn vào chỗ lần thử sai rẽ khỏi lần đúng |
| CAST — 2026 | [2606.18810](https://arxiv.org/abs/2606.18810) | Giữ GRPO, thầy dừng gradient chỉnh advantage từng token; không cần lời giải tham khảo |
| H²SD — 2026 | [2607.18955](https://arxiv.org/abs/2607.18955) | Câu đúng: chỉnh trọng số; câu sai: chưng cất phân phối |
| OGLS-SD — 2026 | [2605.12400](https://arxiv.org/abs/2605.12400) | Lái logit thầy theo kết quả để bớt lệch thầy–trò |
| Respecting Self-Uncertainty — 2026 | [2605.13255](https://arxiv.org/abs/2605.13255) | Dùng entropy của thầy làm tín hiệu thêm |
| DemoPSD — 2026 | [2607.02502](https://arxiv.org/abs/2607.02502) | Điều biến theo độ bất đồng thầy–trò |
| UniSD — 2026 | [2605.06597](https://arxiv.org/abs/2605.06597) | Khung thống nhất cho các cách tự chưng cất |
| π-Play — 2026 | [2604.14054](https://arxiv.org/abs/2604.14054) | Self-play tự tạo thông tin đặc quyền, không cần dữ liệu ngoài |
| Rebellious Student — 2026 | [2605.10781](https://arxiv.org/abs/2605.10781) | Đảo tín hiệu thầy để khuyến khích khám phá |
| SD-Search, AHEAD, GRSD — 2026 | [2605.18299](https://arxiv.org/abs/2605.18299), [2608.24114](https://arxiv.org/abs/2608.24114), [2607.28076](https://arxiv.org/abs/2607.28076) | Cùng ý trên, cho agent tìm kiếm / nhiều lượt |

### Bằng chứng ngược — đọc trước khi chọn hướng

| Paper | ID | Phát hiện |
|---|---|---|
| Why Does Self-Distillation Degrade Reasoning? — Kim et al. 2026 | [2603.24472](https://arxiv.org/abs/2603.24472) | Tự chưng cất nén các token kiểu "chờ đã, kiểm lại", câu ngắn đi, tụt mạnh trên toán ngoài phân phối |
| Rethinking OPSD for Thinking Models — Kaur et al., ICML 2026 | [2607.05184](https://arxiv.org/abs/2607.05184) | Chưng cất ngữ cảnh đặc quyền làm tụt trên chuỗi suy luận dài |
| Rethinking Privileged Information in OPSD — 2026 | [2608.18271](https://arxiv.org/abs/2608.18271) | Điểm số và độ khớp phân phối không đủ để biết thông tin đặc quyền giúp ở đâu |
| Variational Policy Distillation §4.2 (đã có ở individual-project) | [2605.15113](https://arxiv.org/abs/2605.15113) | Mọi cách tự chưng cất sụp với toán và model base |

Một paper khác báo SDPO trên model **base**: điểm về 0 trong 20 bước, độ dài câu bùng nổ.

## Survey

- On-Policy Distillation for LLMs — Song & Zheng 2026, [2604.00626](https://arxiv.org/abs/2604.00626)
- A Brief Overview: On-Policy Self-Distillation in LLMs — 2026, [2605.18141](https://arxiv.org/abs/2605.18141)
- A Formula-Driven Survey and Research Agenda for OPD — 2026, [2606.22793](https://arxiv.org/abs/2606.22793)
- Danh sách cập nhật: <https://github.com/chrisliu298/awesome-on-policy-distillation>
