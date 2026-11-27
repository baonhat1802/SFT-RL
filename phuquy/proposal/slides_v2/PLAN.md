# Plan: làm lại slide proposal theo present-slide

Nguồn: `GStar Proposal — Efficient Reasoning Post-Training.pptx.pdf` (8 trang, xuất từ Google Slides).
Lý do làm lại: các thầy nhận xét slide "trông như AI làm".
Đầu ra dự kiến: `proposal/slides_v2/proposal.html` + `proposal.pdf` (1600×900, xuất bằng `export.py` của present-slide).

---

## 1. Hai skill nói gì, và cách ghép

**present-slide** (`~/working_space/Claude_Skills/present-slide-chis/`, chưa cài vào `~/.claude/skills`, chỉ đọc file):
quy định *hình thức*, và quy định khá ít.

- Một file HTML, mỗi slide là `<section class="slide">` 1600×900, không framework.
- Bố cục: tiêu đề căn giữa ở trên, **một** hình chính ở giữa (biểu đồ, sơ đồ, công thức), tuỳ chọn vài cột ngắn giải
  thích các phần của hình, **một câu thông điệp** màu nhấn ở dưới, dòng nguồn nhỏ, số trang.
- Nền trắng, nhiều khoảng trống, vài màu nhạt cộng một màu nhấn. Font Jost (tiêu đề, nét mảnh) + Source Sans 3.
- Một ý một slide; không vừa thì tách, không thu nhỏ chữ. Giải nghĩa từ viết tắt ở lần đầu.
- `export.py` xuất PDF + PNG từng slide và trả mã 2 nếu chữ tràn khỏi slide.

**make-slides** (`~/.claude/skills/make-slides/`): viết cho beamer, nhưng phần *cách viết và cách trình bày nội dung*
áp dụng được cho mọi định dạng. Đây là phần present-slide không nói. Các quy tắc lấy sang:

| Quy tắc make-slides | Áp vào deck HTML thế nào |
|---|---|
| Viết thông điệp TRƯỚC, rồi mới chọn bằng chứng | Mỗi slide trong §4 có câu thông điệp viết sẵn; hình chọn để chứng minh câu đó |
| Một bằng chứng mỗi slide; không vừa hình vừa bảng cùng số liệu | Bảng 7×7 cũ tách thành 2 biểu đồ, mỗi cái một ý |
| Không em dash; tối đa 2 chỗ in đậm/tô màu mỗi slide; không viết HOA để nhấn | Kiểm bằng grep trước khi xuất |
| Không hiện công thức khi ký hiệu chưa được giới thiệu; slide ký hiệu đi trước; chú thích từng toán tử | Công thức cân bằng gradient tách thành 2 slide (ký hiệu → công thức có chú thích) |
| Nói cái gì là gì TRƯỚC khi báo nó được bao nhiêu điểm | Slide giải thích SFT/GRPO/rollout/advantage đứng trước mọi kết quả |
| Cơ chế, quy trình là **sơ đồ**, không phải gạch đầu dòng; chữ trong ô 3–4 từ | Sơ đồ pipeline viết lại, ô chỉ còn tên ngắn |
| Giữ thuật ngữ tiếng Anh quen thuộc, giải nghĩa một lần | Giữ SFT, RL, GRPO, rollout, advantage |
| Mỗi con số xuất hiện đúng một chỗ; truy được về nguồn; cùng cách làm tròn | Dòng nguồn dưới mỗi slide có số |
| Lưu ý cần để đọc đúng con số phải nằm TRÊN slide, không giấu vào ghi chú | Ví dụ ± là gì, đo ở trần bao nhiêu token |
| Không mục lục kiểu danh sách; slide cuối mang thông điệp, không ghi "Thanks" | Slide cuối = câu hỏi nghiên cứu + nhóm |
| Màu trung tính là sắc nhạt của màu nhấn, không phải một màu be không liên quan | Chọn lại 3 màu nhạt cùng họ với màu nhấn |
| Dấu thập phân thống nhất giữa biểu đồ và chữ | Deck tiếng Anh → dấu chấm |

**Chỗ hai skill khác nhau, và cách chọn:**

- *Tiêu đề*: present-slide dùng nhãn ngắn; make-slides bảo hỏi người dùng. Đề xuất: **tiêu đề là nhãn 2–4 từ, câu thông
  điệp ở dưới là một câu khẳng định đầy đủ**. Khung của present-slide sinh ra để làm vậy, và tiêu đề ngắn ở cỡ 80px
  không bị xuống dòng.
- *Chữ trên slide*: make-slides muốn từ khoá, không câu; bộ quy tắc `writing_skill.md` (audience-first) muốn câu đầy
  đủ. Đề xuất: **câu thông điệp và chữ trong cột là câu ngắn đầy đủ** (người không có mặt vẫn đọc được), nhưng mỗi
  slide tối đa khoảng 40 chữ ngoài hình. Phần giải thích dài đi vào `script.md`, đây là lời nói cho từng slide, thay
  cho `\note{}` của beamer, vì HTML không có ghi chú ẩn.
- *Nền*: make-slides dùng ngà (ivory); present-slide dùng trắng. Theo present-slide (trắng), vì mọi biểu đồ đều tự vẽ
  bằng SVG nên không có ô trắng của hình lấy từ paper.

---

## 2. Vì sao deck cũ trông "AI"

Xem cả 8 trang (ảnh ghép: `.work/slides_old/sheet.png`):

1. **Dòng chữ nhỏ cam, giãn chữ, viết hoa trên mọi tiêu đề** (`THE PROBLEM`, `HOW WE SOLVE`, `EARLY RESULTS`). Đây là
   dấu hiệu rõ nhất của template do AI sinh ra.
2. **Lưới thẻ lặp lại ở gần như mọi slide**: thẻ `01`/`02`, ba thẻ `NOW / THE GAP / NEXT`, ba thẻ tóm tắt dưới bảng,
   ba thẻ nhóm. Người xem nhận ra khuôn từ slide thứ ba.
3. **Câu kiểu khẩu hiệu**: "Balance isn't a decision.", "Rollout-free ≠ compute-free", "a deliberately neutral start".
   Thêm em dash ở tiêu đề slide 9 và trong slide 2.
4. **Chữ dạng đoạn văn trong thẻ**, cỡ nhỏ: chiếu lên màn hình thì không đọc kịp trong lúc nghe.
5. **Công thức có ký hiệu chưa được giới thiệu**: χ, β, τ, H, v_k, "probe", K đều xuất hiện mà không có định nghĩa.
6. **Bảng 7 dòng × 7 cột, mọi ô có ±**, rồi ba thẻ bên dưới đọc lại chính các con số trong bảng (cùng số liệu nói hai lần).
7. **Bảng màu navy + kem + tiêu đề serif đậm**, giống bộ mặc định của các deck do LLM tạo.
8. Lỗi lắp ghép: số trang theo thứ tự 2, 4, 6, 7, 9, 19, 5 (thiếu slide 3, 8, 10–18); chữ `GST AR` bị giãn sai.

---

## 3. Vấn đề nội dung cần xác nhận trước khi làm (cần trả lời)

**Đã trả lời 2026-11-25:** model là Qwen3-0.6B (giữ như bản cũ); số trang sai là lỗi xuất file; ± là độ lệch chuẩn qua
3 seed. Deck chỉ convert phần nhóm đã có; phần cập nhật tuần mới (bài nộp cho Thang Luong) sẽ thêm sau theo hướng dẫn
của người dùng. Bản đã dựng: `proposal.html` / `proposal.pdf`, lời nói ở `script.md`.

**Đổi hướng 2026-11-25 (người dùng):** bỏ PromptSelectRL (thành viên phụ trách đã tách nhóm làm riêng, bỏ khỏi danh sách
tác giả); motivation = ưu/nhược của SFT và RL (có trích dẫn); trọng tâm là "kết hợp ưu điểm của hai cái", không phải điểm
chuyển; phần phương pháp trình bày **dạng tổng quát** (mục tiêu với μ, q, D, λ, w; các phương pháp là trường hợp riêng;
phần mở là bài toán hai tầng). Cân bằng theo gradient của nhóm chỉ còn là một dòng trong bảng trường hợp riêng. Dàn §4
bên dưới là bản cũ, không còn đúng; xem `proposal.html`.

1. **Model trong bảng kết quả**: tiêu đề bảng ghi *"Qwen-3 0.6B, No thinking"*, nhưng plan baseline của repo
   (`PLAN_baselines_qwen25_3b.md`, README) chốt **Qwen2.5-3B**. Bảng đó chạy trên model nào?
2. **Deck đầy đủ**: PDF chỉ có 8 trang, số trang lên tới 19. Có bản đầy đủ không? Slide về "control" thứ ba và về câu
   hỏi hệ thống (tăng tốc do thuật toán hay do engine RL) đang thiếu.
3. **± trong bảng** là độ lệch chuẩn của cái gì (giữa các seed train? giữa các lần lấy mẫu khi chấm?). Phải ghi trên slide.
4. **Có đưa kết quả R01 của mình vào không?** (GRPO @2048 +6 điểm so với model gốc; nhánh thầy-tự-chưng-cất sụp.)
   Mặc định là KHÔNG: đây là proposal của cả nhóm, R01 là hướng riêng chưa được nhóm thống nhất.
5. **Hoàn cảnh trình bày**: bao nhiêu phút, trình bày trực tiếp hay deck còn được gửi đi để đọc riêng (đọc riêng thì
   tách thêm slide, không nhồi chữ). Ngôn ngữ: giữ tiếng Anh như bản cũ?
6. **Thứ tự**: đề xuất chuyển kết quả baseline lên **trước** phần phương pháp, làm bằng chứng cho vấn đề (xem §4).
   Bản cũ để kết quả sau phương pháp.

---

## 4. Dàn slide đề xuất (12 slide, khoảng 15–18 phút)

Mỗi dòng: **tiêu đề** · hình chính · *câu thông điệp* (viết trước). Câu thông điệp tiếng Anh vì deck tiếng Anh.

| # | Tiêu đề | Hình chính | Câu thông điệp |
|---|---|---|---|
| 1 | When to switch from SFT to RL | Bìa: tiêu đề, một dòng câu hỏi, 5 tên | (bìa) |
| 2 | Two ways to learn a solution | Sơ đồ một câu hỏi đi hai nhánh: SFT chép lời giải mẫu; GRPO sinh 8 lời giải, chấm 0/1, so với trung bình nhóm | *SFT learns from a reference solution; GRPO learns only from the student's own attempts.* (định nghĩa rollout, group, advantage, pass rate) |
| 3 | Imitation first costs points | Biểu đồ cột TB 5 bộ @512: chưa train 14.4 · SFT 8.1 · GRPO 21.9 · SFT→GRPO 17.2 | *Starting with SFT costs 4.7 points that the following GRPO stage does not recover.* |
| 4 | Short budgets cut answers off | Đường: TB theo trần 512/1024/2048/4096 (14.4 → 23.4), kèm % bài bị cắt nếu có số | *At 512 tokens the untrained model loses 9 points, mostly from answers cut off before the end.* (cần xác nhận số % bị cắt của nhóm) |
| 5 | Some prompts teach nothing | Đường cong P(mixed \| p) = 1 − p⁸ − (1 − p)⁸, đánh dấu p = ½ | *A question the student always or never solves gives every rollout zero advantage.* |
| 6 | Our pipeline | Sơ đồ 5 ô, chữ ngắn: question pool → prompt selection → SFT branch / GRPO branch → gradient balancing → update. Hai ô đang đề xuất được tô màu nhấn | *We keep one training loop and change two things: which questions get rollouts, and how SFT and RL steps are weighted.* |
| 7 | Predicting pass rate | Ba bước nằm ngang: features (loss, entropy trên lời giải mẫu) → logistic regression → chọn câu gần ½, giữ 10% ngẫu nhiên | *The selector spends rollouts on questions predicted to give mixed results, before generating anything.* Dòng nguồn: chi phí forward + refit tính vào ngân sách GPU-hour |
| 8 | Notation | Bảng nhỏ: L_SFT, L_RL, a, ‖∇L‖ trên probe, v (trung bình trượt), probe = khối decoder cuối | *Both losses are measured on the same small part of the network.* |
| 9 | Balancing by gradient size | Công thức a = clip(v_SFT / (v_SFT + v_RL), 0.1, 0.9), chú thích dưới từng phần | *The loss with the larger gradient gets the smaller weight, so neither dominates an update.* KL/entropy và luật "không có nhóm có phương sai → chỉ SFT" ghi ở dòng nguồn hoặc script |
| 10 | What we compare against | Bảng 3–4 dòng ngắn: GRPO only · SFT → GRPO (điểm chuyển chỉnh trên dev) · trộn tĩnh 0.5 : 0.5 · bản đề xuất; cột: cùng ngân sách GPU-hour | *Every method gets the same GPU-hour budget, so a gain cannot come from extra compute.* |
| 11 | Open question: when to lean on SFT | Sơ đồ nhỏ: tín hiệu đang log (tỉ lệ nhóm toàn sai / nhóm lẫn) → nghiêng SFT / nghiêng RL | *Balancing is neutral today; the next step is a schedule driven by signals we already log.* |
| 12 | Team | Ba cột: Balancing (Bao Nhat, Huy Hoang) · RL engine (Hong Son, Phu Quy) · Prompt selection (Munchic) + câu hỏi nghiên cứu nhắc lại | *Câu hỏi nghiên cứu một câu*, không ghi "Thanks" |

Ghi chú về dàn ý:

- Hai "vấn đề" của slide 2 cũ được tách thành bằng chứng riêng (slide 3–5) thay vì hai thẻ chữ.
- Bảng 7×7 cũ bị bỏ khỏi thân deck; con số đầy đủ để ở slide phụ lục cuối (nếu cần trả lời câu hỏi), mỗi số vẫn chỉ
  xuất hiện ở đó và trong một biểu đồ.
- "Three controls" của bản cũ: chỉ thấy hai control trong 8 trang (PromptSelect, Smart balancing). Chờ câu 3.2 ở §3.

---

## 5. Cách làm

1. **Chốt §3 và dàn ở §4** với bạn. Chưa viết HTML trước bước này.
2. Dựng `proposal.html` từ khung CSS của `example/example.html` (giữ nguyên lớp `.slide`, `.take`, `.src`, `.pg`).
   Thêm lớp cho: sơ đồ SVG, biểu đồ SVG, bảng ký hiệu, công thức có chú thích.
3. **Biểu đồ vẽ bằng SVG nội tuyến**, số liệu lấy từ một chỗ duy nhất (`numbers.json` cạnh deck) để mỗi con số có
   một nguồn. Không chép ảnh từ PowerPoint cũ.
4. Viết `script.md`: lời nói cho từng slide (phần mà make-slides để trong `\note{}`).
5. Xuất: dùng env `gstar2026` (không cài vào `base`):
   ```bash
   conda run -n gstar2026 uv pip install playwright
   conda run -n gstar2026 python -m playwright install chromium
   conda run -n gstar2026 python ~/working_space/Claude_Skills/present-slide-chis/export.py proposal/slides_v2/proposal.html
   ```
   Mã thoát 2 = có chữ tràn → sửa bằng cách tách hoặc cắt chữ, không thu nhỏ font.
6. **Kiểm trước khi đưa bạn xem**:
   - `export.py` báo `ok: no slide overflows`.
   - Ghép PNG thành một ảnh và nhìn từng slide (export không bắt được hình quá nhỏ, khoảng trống giữa slide, cột bị cắt).
   - `grep` em dash (`—`), `≠`, chữ viết hoa toàn bộ; đếm chỗ tô màu mỗi slide (≤ 2).
   - Quét ký hiệu: liệt kê mỗi ký hiệu và slide đầu tiên nó xuất hiện; phải sau slide 8 (ký hiệu).
   - Quét tên riêng (PromptSelectRL, probe, advantage…): slide định nghĩa phải đứng trước slide dùng nó.
   - Mỗi con số khớp nguồn, cùng cách làm tròn.
7. Gửi bạn ảnh các slide sau từng đợt (khoảng 4 slide), sửa theo góp ý, không đợi xong cả deck mới cho xem.
