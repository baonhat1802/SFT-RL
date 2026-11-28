# Trạng thái capstone — đọc trước tiên

Mục tiêu (người dùng chốt 2026-11-21): **vượt các baseline trong bảng của nhóm** trên dữ liệu curated, trần 512 token. Baseline mạnh nhất
của bảng là **GRPO-only** (full FT 22,3 ± 1,5; LoRA 21,9 ± 0,8), không phải SFT→GRPO (17,0–17,2): SFT trên lời giải Luna làm model tụt
14,4 → 8,1, nên mọi cách đi qua SFT-Luna đều bị kéo xuống. Quy trình vòng ý tưởng: skill `cs-brainstorm` → `cs-novelty-gate` → `cs-round`
→ `cs-leaderboard` (bắt chước Peer_reconstruction).

## Bảng của nhóm (do bạn cùng nhóm chạy; code và cách chấm chưa có trong repo)
| Train | Ghi chú | Seq | AIME24 | AMC23 | MATH500 | Minerva | Olympiad | TB | % cắt |
|---|---|---|---|---|---|---|---|---|---|
| Base | | 512 | 0,0 | 10,0 | 40,5 | 12,4 | 9,0 | 14,4 ± 0,5 | 66,6 |
| Base | | 1024 | 1,1 | 25,0 | 50,2 | 14,0 | 18,5 | 21,8 ± 1,0 | 28,6 |
| Base | | 2048 | 4,4 | 25,0 | 50,8 | 14,1 | 20,4 | 22,9 ± 0,6 | 8,2 |
| Base | | 4096 | 2,2 | 27,5 | 52,7 | 14,2 | 20,5 | 23,4 ± 1,3 | — |
| SFT | LoRA r32 α64, lr 1e-4, 3 ep, batch 64 | 512 | 1,1 | 5,8 | 23,0 | 6,0 | 4,6 | 8,1 ± 1,0 | 2,3 |
| GRPO | LoRA r32 α64, lr 1e-5, ckpt-1400 | 512 | 2,2 | 23,3 | 51,7 | 14,6 | 17,8 | 21,9 ± 0,8 | 21,3 |
| SFT→GRPO | LoRA, ckpt-2004 | 512 | 0,0 | 13,3 | 45,5 | 12,7 | 14,7 | 17,2 ± 1,1 | 42,7 |
| GRPO | full FT, lr 1e-6, ckpt-1800 | 512 | 0,0 | 23,3 | 53,1 | 15,3 | 19,7 | 22,3 ± 1,5 | 14,3 |
| SFT→GRPO | full FT, ckpt-1400 | 512 | 2,2 | 16,7 | 43,7 | 9,4 | 12,8 | 17,0 ± 2,2 | 55,3 |

Đọc bảng: GRPO-only@512 (22,3) ≈ Base@2048 (22,9) — phần lớn lợi ích GRPO là **viết gọn để khỏi bị cắt** (66,6% → 14,3%).

## Cách chấm của mình vs của nhóm (điểm neo)
`code/beat/eval_bridge.py`: 5 bộ của `code/data/bridge/eval_math`, prompt SimpleRL (curated), nhiệt độ 0,6, n = 4 mẫu/câu, Math-Verify,
"±" = độ lệch chuẩn giữa các chỉ số mẫu. Model gốc, 512 token **chỉ tính phần trả lời**: TB 17,0 ± 0,8, cắt 59,4% (smoke, n = 2) —
**lệch bảng nhóm +2,6**. Giả thuyết: "Seq len 512" của nhóm tính cả prompt → đang thử (`--budget-includes-prompt`). Khớp thì so thẳng với
bảng nhóm; không khớp thì chỉ so trong pipeline của mình (xin checkpoint của nhóm để đánh giá lại, ~5 phút/model).

## Hạ tầng
- **VM đang dùng (từ 2026-11-26): `phuquy-a100`** trong project NHÓM `team-10-2026-99999`, zone **us-central1-b**, a2-highgpu-1g (1× A100
  40GB, 12 vCPU, 83 GB RAM), ổ 200 GB, nhãn owner=phuquy. ~3,67 USD/giờ khi chạy — **tắt khi xong** (đã tắt 26/11 sau kiểm trước R02-1).
  Vào: `gcloud compute ssh phuquy-a100 --zone us-central1-b --project team-10-2026-99999`. Trong project còn VM `hoang` (A100 80GB) của
  bạn cùng nhóm — không đụng.
- Trên VM: `~/miniconda3/envs/capstone` (vllm 0.15.1, trl 0.22.2, transformers 4.57.6, torch 2.9.1 cu128), repo `~/capstone` (bản
  `git archive HEAD` + `code/data/{tracks/curated, bridge/eval_math, bridge_curated_teachers, bridge_curated_teachers_fixed}`), HF `~/hf`.
  Dựng lại: `.work/vm/setup_team_vm.sh` (+ `install.sh`), ~4 phút; gói code+dữ liệu 18 MB + 9 MB.
- **VM thứ hai và thứ ba (27/11):** `phuquy-a100-b` (dựng từ `vm_payload.tgz` cũ + gửi bù code/dữ liệu) và `phuquy-a100-c` (dựng từ
  **`.work/vm/vm_payload_v2.tgz`** — bản đủ: code/beat mới, train, evaluation, scripts, tests, data/{bridge, bridge_curated_teachers(_fixed),
  tracks}, research/rounds; 84 MB, đã quét token; script `.work/vm/setup_vm_v2.sh`). Cùng cấu hình a2-highgpu-1g, us-central1-b. Để ở trạng
  thái TẮT, lần sau `gcloud compute instances start <tên>` là dùng được. Tắt rồi vẫn tốn tiền ổ đĩa: 200 GB pd-balanced ≈ 20 USD/tháng mỗi máy.
- VM cũ `capstone-a100-v2` (project cá nhân `nguyenlamphuquykh-prj`) không dùng được: **billing của project cá nhân đã tắt** (26/11), kể cả
  snapshot `capstone-snap-1121`.
- Quota nhóm: A100 40GB 16 card/vùng; A100 80GB 1 (VM hoang đang dùng); **không có H100/H200/B200**; L4 chậm hơn A100.
- Dừng run trên VM: dọn cả `VLLM::EngineCore` mồ côi theo PID trên GPU (`nvidia-smi --query-compute-apps=pid`); `pkill -f` phải dùng mẫu
  không tự khớp chính lệnh ssh (vd `"python -m beat.diag_[c]os"`).
- Không gửi file lớn từ Mac (mạng tải lên chậm). Chạy nền bằng `setsid nohup … &`.

## Nhật ký
- 2026-11-21: dựng VM, cài env; viết `code/beat/` (eval_bridge, gen_sft_data, phase1.sh); thêm tham số tuỳ chọn cho `train/sft.py`
  (`--train-file --lr --lora-r --lora-alpha --run-name`; mặc định giữ nguyên). Smoke phase 1 chạy hết đường: sinh dữ liệu có gợi ý phủ 80%
  câu (đúng 46,5% mẫu) vs tự giải 61% (đúng 36,3%) trên 64 câu. Đang chạy: điểm neo cách chấm, rồi phase 1 đầy đủ (SFT trên dữ liệu
  gợi ý / tự giải / Luna, đánh giá @512).
- 2026-11-21: dựng quy trình vòng ý tưởng (skill cs-*, `research/`). Hướng người dùng đang nghĩ: tối ưu hai tầng kiểu BRIDGE nhưng thầy là
  tự chưng cất on-policy (model có `y*` trong ngữ cảnh); dạng max–min có sẵn qua phép phạt của BRIDGE (công thức trong hội thoại
  2026-11-21, chưa ghi vào `proposal/`). Phải qua `cs-novelty-gate` (G2c: ghép BRIDGE + OPSD) trước khi cài.
- 2026-11-21: điểm neo cách chấm KHÔNG khớp bảng nhóm (512 chỉ câu trả lời: 17,0; tính cả prompt: 12,9; bảng: 14,4) → chỉ so trong
  pipeline của mình. Model gốc @512 (n = 4): TB 16,8 ± 1,0, cắt 60,0%. Smoke phase 1: 3 bước SFT-Luna đã 17,0 → 13,0.
- 2026-11-21: vòng R01 (6 ý tưởng, 12 run, `research/rounds/R01/`) chạy trên khung riêng `code/beat/loop.py` (Unsloth vá TRL quá sâu để
  chen ý tưởng). Meta-OPSD đổi sang kiểu MAML (bậc hai qua sai phân) theo người dùng. Chuỗi trên VM: `~/capstone/chain_r01.sh`
  (phase 1 → smoke R01 → run_jobs R01 → GRPO bản Unsloth làm điểm neo). Log: `~/chain_r01.log`, `~/capstone/runs/r01/`.
- 2026-11-21: **phase 1 xong** (GCP, kết quả ở `experiments/phase1/`, @512 n = 4, TB 5 bộ / DEV / % cắt): model gốc 16,8 / 23,6 / 60,0;
  SFT "gợi ý rồi tự viết" 17,7 / 23,8 / 53,2; **SFT "tự giải, giữ bài đúng" 19,6 / 26,5 / 53,6**; **SFT Luna 12,7 / 16,4 / 2,4**. SFT-Luna tụt
  như bảng nhóm (cắt 2,4% ≈ 2,3% của nhóm) → giả thuyết lệch văn phong đứng vững trong pipeline của mình. Bất ngờ: dữ liệu có gợi ý
  kém dữ liệu tự giải dù phủ nhiều câu hơn (54% vs 46%) → dấu hiệu "imitation gap" (lời giải viết khi biết đáp án có bước nhảy trò không tái
  tạo được) — rủi ro cho các ý tưởng OPSD. Mốc model gốc @2048 (cụm): 26,4 / 33,9 / 5,2. Máy GCP tự tắt sau phase 1.
- 2026-11-21: R01 chuyển sang train @2048 (người dùng). metaopsd: thêm `hess_clip=1` (số hạng bậc hai lên 60× gradient GRPO ở smoke).
  Sự cố: dừng một run bằng pkill để lại `VLLM::EngineCore` mồ côi giữ card → các run sau trên card đó chết dây chuyền; dọn bằng
  `.work/vm/kill_w1.sh` (giết theo PID trên GPU). Bài học: dừng run thì giết cả EngineCore trên đúng GPU.
- **2026-11-26 — tối ưu code (`code/beat/fastlp.py`, dùng trong `loop.py`, `diag_cos.py`).** Ghép micro-batch theo độ dài dưới ngân sách
  4.096 token + log-prob theo khúc có checkpoint (không giữ logits fp32) + vLLM ngủ lúc train. Kiểm (`bench_fastlp.py`): từng mẫu khớp code
  cũ tới 2e-6; khi ghép batch nhiễu bf16 ngang code cũ (gradient cos với mốc 0,9955 vs 0,9965). forward+backward ×1,97, bộ nhớ đỉnh 9 vs 15 GB.
  Đầu-cuối GRPO @2048: 332 → 247 s/rollout (×1,34), rollout 0 phần thưởng trùng. Sinh lời giải chưa nhanh hơn (nút thắt có lẽ `max_num_seqs`).
  `--tok-budget 0 --vllm-sleep 0` = đường cũ.
- **2026-11-26 — chẩn đoán vì sao Meta-OPSD sụp (`code/beat/diag_cos.py`, `runs/diag_cos` + `runs/diag_tokw` trên VM).** Tại model gốc,
  cos(gradient hướng dẫn trên nửa câu A, gradient GRPO trên nửa câu B), 3 batch × 256 câu: mốc GRPO-vs-GRPO +0,15 (0,07–0,29, rất nhiễu);
  thầy ngắn gọn trên nhóm cả-sai (cấu hình R01) +0,06; thầy prompt đầy đủ +0,16 (dương 3/3); thầy trên lời giải sai của nhóm lẫn +0,17;
  lời giải đúng +0,05; chỉ đáp án +0,01; bỏ token từ `\boxed` +0,04; RLSD (trọng số từng token) không thêm tín hiệu (+0,08 vs +0,10 ở
  trọng số 1; [1;1] khớp mốc tới 4 chữ số → đo đúng). **cos −0,27 trong log R01 là ẢO** do đo tại θ′ sau bước trong: bước theo chính
  gradient GRPO cũng tụt ~0,25 khi đo tại θ′ (3/3 batch). Kết luận: thầy không ngược phần thưởng ở điểm xuất phát; Meta-OPSD sụp do bộ máy
  cập nhật (bước thầy norm cố định 0,1 lấn bước GRPO ~0,08; gradient tại điểm dịch; Hessian sai phân nhiễu). Prompt thầy ngắn gọn (đổi 21/11)
  và chọn nhóm cả-sai làm yếu tín hiệu. Mọi chỉ số cos trong log R01 không dùng để chẩn đoán được.
- **2026-11-26 — vòng R02 (người dùng: "cần novelty hơn", không GRPO + λ·chưng cất).** Rà văn liệu: thầy tương phản, thầy bộ nhớ, đồng thuận
  nhiều thầy, câu hỏi con từ y*, RLSD đều đã có (`prior_art.md`). 2 ý tưởng qua lọc (`research/rounds/R02/ideas.md`). **R02-1 rẽ nhánh tại
  chỗ thầy-có-y* phản đối CHẾT ở kiểm trước** (`code/beat/diag_fork.py`, 262 câu cả-8-sai, k = 8): sinh lại từ đầu cứu 23,7%, vị trí ngẫu
  nhiên cùng độ sâu 24,0%, thầy tốt nhất 21,4%, thầy chỗ phản đối đầu tiên 18,3%; rẽ càng sâu càng kém (entropy 11,1%). Phát hiện phụ: ~24%
  câu "cả-8-sai" được cứu chỉ bằng 8 mẫu nữa. Người dùng chọn: brainstorm họ khác (không phải "thầy-có-y* dạy trò"). VM đã tắt.

- **2026-11-26 — thầy trên model base có "đọc" y* không? (`diag_cos.py --format-sweep`, `runs/diag_fmt/batches.jsonl`, 2 batch × 256
  câu; dừng sau 2 batch để tiết kiệm).** 6 cách trình bày y* × 2 loại mẫu. |log π_T − log π_S| trung bình chỉ 0,04–0,11 nat/token, 0,3–1,9%
  token đổi > 1 nat với MỌI cách → thầy base gần như không phản ứng với y*. Chuyển giao (mốc GRPO 0,03 / 0,12): ngắn gọn và đầy đủ dương ở
  cả 2 batch (0,02–0,23); chỉ đáp án ≈ 0; `multiturn` lẫn dấu; `system` âm; `selfprefix` (y* nằm sẵn đầu câu trả lời) đổi thầy MẠNH nhất
  (1,9% token) nhưng chuyển giao ≤ 0. → Ép thầy base phản ứng mạnh hơn không làm tín hiệu tốt hơn; dạng chỉ dẫn hiện tại là tốt nhất trong
  6 dạng. Nhánh "định dạng ngữ cảnh cho model base": âm tính. Người dùng: bỏ nhánh kiểm phép đo bằng 4 run train. VM đã tắt.
- **2026-11-27 — R02-3 thầy "y* trong trọng số" (một bước SFT trên y* cho từng câu, rồi chấm lời giải của trò) CHẾT ở kiểm trước**
  (`diag_cos.py --adapted-sweep`, `runs/diag_wt`, 2 batch × 256 câu). Mốc GRPO +0,14 (ổn định 0,12–0,16). Gradient SFT trên y* (hướng
  "SFT-Luna") **−0,18** (−0,23; −0,13) — xác nhận bậc nhất kết quả phase 1 (SFT-Luna 16,8 → 12,7). Thầy-trọng-số: ε = 0,3 / 1 / 3 làm 0,7% /
  21% / 89% token đổi > 1 nat (thầy-ngữ-cảnh 1,4%) nhưng chuyển giao luôn âm (−0,16 … −0,22) — kế thừa hướng có hại của SFT trên y*.
  Thầy-ngữ-cảnh prompt đầy đủ +0,15 trên cả hai loại mẫu (ổn định); ngắn gọn ≈ 0. VM đã tắt.
- **2026-11-27 — Vì sao SFT trên y* chống phần thưởng** (`diag_cos.py --ystar-split`, `runs/diag_ystar`, 2 batch × 256 câu, kết quả
  `experiments/rounds/R02/diag_ystar_results.json`). Thống kê dữ liệu: y* trung vị 115 token (lời giải đúng tự sinh 356–489). Tách gradient SFT
  trên y* theo loại token (mốc GRPO +0,08): toàn bộ −0,15; chữ (26% token, NLL 2,03/token) −0,13; ký hiệu toán (46%, NLL 0,74) −0,12; số trong
  công thức (18%, NLL 0,41) −0,06; `\boxed` ≈ 0; y* ngắn/dài −0,15/−0,14. SFT trên lời giải đúng của chính trò +0,07; cos(SFT-tự-thân, SFT-y*)
  −0,12. → Giả thuyết "nhảy bước/số bịa" SAI (số dễ đoán, gần trung tính); phần gây hại là GIỌNG VĂN của Luna (chữ + cách viết ký hiệu), và
  hướng đó ngược với hướng bắt chước chính mình. Che token không cứu được (mọi loại đều ≤ 0) → bỏ ý "chỉ bắt chước kế hoạch". Khớp DASD
  2606.00628, 2603.24472 (lệch giọng văn khi bắt chước văn bản có tham chiếu). Lỡ chạy model trên Mac trước đó (lag máy) — đã dừng; từ nay chỉ
  chạy trên VM. VM đã tắt (~25 phút).
- **2026-11-27 — Kiểm trước vòng R03 (nhân quả)** (`diag_cos.py --causal-sweep`, `runs/diag_causal`, 2 batch × 256 câu, seed 17; kết quả
  `experiments/rounds/R03/diag_causal_results.json`). Mốc GRPO +0,08 (0,074; 0,087).
  R03-1 cặp phản thực tế cùng giọng: y* trừ bản đổi 1–2 giá trị tính ra **+0,105** (0,096; 0,114) — QUA; đổi dấu ± +0,05 (0,02; 0,08);
  đối chứng đảo dòng −0,04 (neo đạt); cos với SFT-y* −0,05 (hướng mới). R03-2 AND-mask: 2 giọng +0,094 = trung bình không mask +0,093 (mask
  ngẫu nhiên cùng mật độ +0,069); 3 môi trường +0,089, batch 1 dưới mốc → KHÔNG QUA (mask không thêm gì so với cộng thường). Placebo: thầy thật
  +0,06, thầy với y* của câu khác +0,046, hiệu +0,024 → phần lớn tín hiệu thầy-ngữ-cảnh đến từ NGỮ CẢNH, không từ nội dung y*.
  **Sửa nhận định:** chuyển giao của SFT-y* dao động mạnh giữa các batch: −0,23, −0,13 (diag_wt), −0,17, −0,13 (diag_ystar), +0,03, +0,07
  (diag_causal), trung bình −0,09 — âm ở 4/6 batch, không phải luôn −0,15. Thầy-ngữ-cảnh cũng dao động (+0,15 trước, +0,06 lần này). VM đã tắt.
- **2026-11-27 — R03-1 `cfpair_s1` SỤP, dừng ở rollout 30/66** (theo ngưỡng đã thống nhất: phần thưởng tụt rõ nhiều rollout liền). Log
  `experiments/rounds/R03/cfpair_s1/`. Theo pha (cfpair / grpo_s1): rollout 0–9 phần thưởng 0,359/0,358; 10–19 0,518/0,507 (nhỉnh hơn, nhưng cắt
  6,0% vs 2,1%); 20–24 0,483/0,515; 25–29 **0,403/0,543**, allfail 0,30/0,19. `cf_margin` 0,19 → 0,38 → 0,75 → 1,08 (tăng không chặn); độ dài
  giảm còn ~550–570 (grpo ~710–730) trong khi tỉ lệ cắt vẫn cao hơn → lời giải phân hoá (ngắn hơn, nhưng một phần lặp tới trần). Đọc: số hạng
  −NLL(ỹ*) không chặn đẩy xác suất bản sai xuống mãi và làm hỏng model; tín hiệu bậc nhất tốt ở θ0 (+0,10) không giữ được khi tích luỹ. Nếu thử
  lại: dạng có chặn (logistic/hinge trên margin) và λ nhỏ hơn. Chi phí ~2,5 giờ VM (~9 USD). VM phuquy-a100 đã tắt.
- **2026-11-27 — VM thứ hai `phuquy-a100-b`** (cùng cấu hình, project nhóm, us-central1-b) tạo để chạy song song kiểm trước R04 (doubly robust,
  CATE). `vm_payload.tgz` cũ THIẾU `fastlp.py`, `diag_*.py` và `code/data/bridge_curated_teachers/` → phải gửi thêm `beat_code.tgz` + `bct2.tgz`.
- **2026-11-27 (chiều)** — `diag_opening` xong — câu mở đầu ~64 token của model có-y* chỉ mang 4,5% hiệu ứng của y* (+0,025 vs +0,56);
  câu đầu Luna ép làm đầu lời giải HẠI −0,055 (giọng cụt). Giả thuyết cơ chế thưa ở đầu lời giải bị bác (`research/rounds/R04/ideas.md`).
  `cfpair_reg_s1` (DPO-chặn, λ 0,2, β 10) qua vùng sụp: 32/66 rollout, reward +0,014 ± 0,004 so với grpo_s1 (10 rollout cuối +0,022),
  nhưng lời giải NGẮN hơn rõ (~570 vs ~720 token) → phải xem eval @2048 để loại khả năng lợi chỉ do viết gọn.
- **2026-11-27 (chiều)** — `diag_mediation` xong: phần hiệu ứng y* chuyển qua tiền tố tăng ĐỀU (2% @16 tok, 6% @64, 18% @50%, 31% @75%),
  không có bước then chốt; 75% lời giải có-y* ĐÚNG chỉ được viết tiếp đúng 40% (lời giải thường đúng: 69%) → văn bản có-y* không tự đứng
  được, phần lớn hiệu ứng là chép kết quả ở cuối. Chạy tiếp: `diag_latent` (cấy trạng thái ẩn, VM c) và `diag_cos --skeleton-sweep`
  (khám phá đồ thị nhân quả các bước của y*, VM b). Sự cố: hai script theo dõi đọc nhầm log cũ → VM b bị tắt giữa chừng, VM c tắt trước
  khi chạy bước kế; đã sửa (log mới mỗi lần chạy, chỉ grep một file).
- **2026-11-27 (tối)** — R05-A khung nhân quả của y* (`diag_cos --skeleton-sweep`) LOẠI: khung lọc sạch chữ thường nhưng cos với GRPO
  (bỏ vùng đáp án) −0,065 ≈ bước ngẫu nhiên −0,049; cả y* −0,077; chỉ đáp án +0,026. Bước tính toán của Luna cũng ngược chiều GRPO.
  VM b đã tắt. `diag_latent` chạy lại bản gọn (6 cấu hình, 2 mẫu, 1024 token) vì vòng sinh HF chậm (22 phút/cấu hình ở bản đầy đủ).
- **2026-11-27 (tối)** — `cfpair_reg_s1` (DPO-chặn, λ 0,2, β 10) eval: @512 TB **21,6** / DEV 28,6, cắt 58,6% (grpo_s1: 17,6 / 23,8, cắt 68,8%) →
  **+4,0 TB ở trần protocol 512**; @2048 TB 30,9 / DEV 39,5 (grpo_s1: 32,4 / 39,9) → −1,5. Lợi thế đến từ viết gọn (học được từ y* Luna) —
  người dùng: viết gọn là hành vi học được, vẫn tính là thắng ở 512. Còn thiếu: seed 2–3, ablation bỏ cặp phản thực tế (chỉ giữ số hạng kéo về y*).
  Log + summary: `experiments/rounds/R03/cfpair_reg_s1/`. Đọc qua snapshot đĩa VM a (vùng hết A100), đĩa tạm/snapshot/máy tạm đã xoá.
- **2026-11-27 (khuya)** — Ablation `cfpair_pos_s1` (giữ lực kéo về y* có chặn, BỎ ỹ*: `--opt neg=0`) SỤP, dừng ở rollout 9/66 theo ngưỡng
  đã thống nhất: phần thưởng 0,27 → 0,14 trong khi reg/grpo 0,40–0,42; độ dài 636 → ~390 token. → lực đẩy ỹ* là THIẾT YẾU: trong cfpair_reg,
  hai lực y*/ỹ* triệt tiêu ở mọi token chung, chỉ còn tác dụng ở token bị đổi giá trị (gnorm 0,03 vs 0,11 khi bỏ ỹ*). Lợi thế +4 TB @512 của
  cfpair_reg KHÔNG phải do "học giọng gọn của Luna" (bản chỉ học giọng gọn thì sụp). Lần đầu VM a: vLLM `illegal memory access` (card), chạy
  lại trên VM b. Log: `experiments/rounds/R03/cfpair_pos_s1/`.
- **2026-11-27 (khuya)** — Ablation `cfpair_nat_s1` (DPO cùng công thức nhưng CẶP TỰ NHIÊN: y* chọn vs một lời giải SAI của chính model)
  SỤP, dừng ở rollout 6/66: phần thưởng 0,26 → 0,18 (grpo 0,37), độ dài 643 → 467, allfail 0,45 — giống `cfpair_pos_s1`. → công thức DPO
  không phải nguồn lợi thế; CÁCH DỰNG CẶP (phản thực tế tối thiểu, giữ văn, chỉ đổi giá trị) mới là. Log `experiments/rounds/R03/cfpair_nat_s1/`.
  Xếp tiếp: placebo `cfpair_plc_s1` lên VM a; VM c: bộ kiểm (gộp) → khoảng cách thầy–trò → `grpo_s2` (mốc cho `cfpair_reg_s2`, VM b).
- **2026-11-28 (sáng)** — `cfpair_reg_s2` (seed 2) eval: @512 TB **21,1** / DEV 28,9, cắt 56,2%; @2048 TB **32,1** / DEV 38,4. Lặp lại seed 1
  (21,6 / 28,6 @512; 30,9 / 39,5 @2048). So grpo_s1 (17,6 / 23,8; 32,4 / 39,9): +3,5…+4 TB @512, ≈ ngang @2048. Mốc cùng seed `grpo_s2` đang
  chạy (VM c). Thí nghiệm khoảng cách thầy–trò (0,5B/1,5B/3B) và bộ kiểm ORM ghi ở `research/rounds/R05/ideas.md`. Đẩy tiếp `cfpair_reg_s3` lên VM b.
- **2026-11-28 (sáng)** — Placebo `cfpair_plc_s1` (cặp đổi chỗ 1–2 từ thường, giữ tính đúng toán): @512 TB **19,0** / DEV 27,1, cắt 58,8%;
  @2048 TB 30,2 / DEV 37,1. Nằm GIỮA grpo_s1 (17,6 / 23,8; 32,4 / 39,9) và cfpair_reg (21,6–21,1 / 28,6–28,9; 30,9–32,1 / 38,4–39,5).
  → lợi thế @512 của cfpair_reg ≈ +1,4 do "có tương phản trên y*" (kéo viết gọn, placebo cũng cắt 58,8%) + ≈ +2,3 do tương phản TÍNH HỢP LỆ
  (chênh chủ yếu AMC 18,1 vs 12,5; Minerva 21,1 vs 17,2). @2048 placebo thua cả hai. Placebo mới 1 seed → đẩy `cfpair_plc_s2` lên VM a.
- **2026-11-28 (sáng)** — Mốc cùng seed `grpo_s2`: @512 TB 17,2 / DEV 23,9, cắt 67,1%; @2048 TB 32,0 / DEV 39,1. → cfpair_reg vs grpo CÙNG SEED:
  seed 1 +4,0 TB / +4,8 DEV @512; seed 2 +3,9 TB / +5,0 DEV @512; @2048 seed 2 +0,1 TB / −0,7 DEV (seed 1 −1,5 / −0,4). Đẩy `grpo_s3` lên VM c
  (cặp với `cfpair_reg_s3` trên VM b). Placebo seed 2 trên VM a.
- **2026-11-28 (trưa)** — Thêm seed: `cfpair_reg_s3` @512 TB **18,5** / DEV 26,9, @2048 31,8 / 38,8; placebo `cfpair_plc_s2` @512 TB **20,6** /
  DEV 28,3, @2048 30,2 / 37,7. Tổng hợp (TB @512 / @2048): grpo 17,6–17,2 / 32,4–32,0 (s3 đang chạy); cfpair_reg 21,6–21,1–18,5 (TB 20,4) /
  30,9–32,1–31,8 (31,6); placebo 19,0–20,6 (19,8) / 30,2–30,2. → @512 placebo GẦN NGANG reg (chênh ~0,6 TB, trong nhiễu) — lợi thế @512 chủ yếu
  do "có cặp tương phản trên y*" kéo viết gọn; ở @2048 reg hơn placebo ~1,4 TB ở mọi seed (placebo làm mất độ đúng, reg thì không). Sửa câu
  "+2,3 TB do tính hợp lệ" (1 seed) thành: tính hợp lệ BẢO TOÀN độ đúng (thấy ở @2048), còn lợi @512 phần lớn đến từ viết gọn.
- **2026-11-28 (chiều)** — `diag_twins` (song sinh dưới can thiệp gợi ý đặc quyền, không train; 384 câu → 120 câu khó ≤2/8, 8 seed ghép
  nhiễu, gợi ý = bước y* đã che số): 76/120 câu khó có bộ gợi ý đủ (PS ≥ 0,25 trên seed gốc sai): 59 chỉ cần **1** bước, 16 cần 2, 1 cần cả;
  44 không bộ nào đủ. Cạnh hiệp đồng 29/484. 257 cặp song sinh, nhưng **rẽ rất sớm**: trung vị 4 token (0,7% bài gốc), 26% rẽ ngay token 0
  → gợi ý ở prompt đổi câu mở đầu, cặp KHÔNG tối thiểu. Hướng gradient (69 cặp): song sinh vs GRPO cos +0,15 (mốc GRPO nửa A–B +0,26),
  cặp tự nhiên vs GRPO −0,31. Lưu ý: chưa hiệu chỉnh đa kiểm định (≈20 bộ/câu, 2/8 lật có thể do may) → bản `diag_causal` sửa bằng BH +
  placebo. Kết quả `.work/vm/out_twins/results.json`.
- **2026-11-28 (chiều)** — Viết `code/beat/diag_causal.py` (kiểm D1–D4 cho khung "dịch chuyển cơ chế thưa + khám phá bằng can thiệp + phản
  thực"): D1 bộ đủ tối thiểu (BH q=0,1 so tỉ lệ đúng gốc, placebo làm mốc dương tính giả), D2 đồ thị hiệu ứng tổng i→j (McNemar ghép seed,
  placebo), D4 quan sát vs can thiệp (Spearman + trần nhiễu nửa-nửa), D3 SCM Gumbel-max chính xác trong HF (nhiễu theo vị trí token của trò,
  chèn ghi chú giữa bài không dịch nhiễu; mốc: chèn rỗng phải tái tạo y hệt, ghi chú placebo). Chạy trên VM d.
- **2026-11-28 (trưa)** — `diag_causal` bản đầy đủ (gợi ý CHE SỐ = do(nước đi); 120 câu khó + 60 câu lưng chừng, 8 seed ghép nhiễu, trung bình
  2,9 bước/câu sau lọc rò): **SỬA kết luận `diag_twins`**: trên câu khó, gợi ý một bước lật lần sai thành đúng 10,8%, placebo (bước của câu
  khác) 8,8%, gốc 9,8%; 2 bước 8,7%, 4 bước 6,0% → gợi ý che số gần như KHÔNG tác dụng trung bình; "76/120 có bộ đủ" của twins phần lớn là
  may (≈20 bộ/câu, ngưỡng 2/8). Sau hiệu chỉnh BH: chỉ 9/120 câu khó có bộ đủ thật (5 cần 1 bước, 4 cần 2), placebo 0/120 qua → D1 không
  đủ mẫu để kết luận về "dịch chuyển thưa". Câu lưng chừng: 1 bước 46,5% vs placebo 40,4% (+6). D2: 0 cạnh qua BH (249 kiểm); hiệu ứng TB
  +5,8 điểm % vs placebo +1,6; trò tự hoàn tất phép tính của nước đi được cho +10,5. **D4 QUA**: tương quan quan sát "trò viết ra số trung
  gian của Luna" ↔ đúng = +0,235, hiệu ứng can thiệp chỉ +0,079; Spearman giữa hai cách đo 0,005 (trần nhiễu 0,34); 27/52 bước "có vẻ quan
  trọng" theo quan sát có hiệu ứng can thiệp ≤ 0. Lưu ý: hai cách đo không hoàn toàn cùng biến (quan sát = số được viết ra; can thiệp = nước
  đi che số). D3 đang chạy (chỉ 5 câu có bước đủ thật). Xếp tiếp bản KHÔNG che số (`--mask 0`, do(sự kiện)) trên VM d.
- **2026-11-28 (trưa)** — D3 (SCM Gumbel-max chính xác; 5 câu khó có bước đủ thật, 28 bài sai × 4 điểm cắt): mốc chèn rỗng tái tạo
  **100%** y hệt → phản thực cài đúng. Chèn bước của Luna ở ĐẦU bài: lật 32% (placebo 0%) nhưng cặp rẽ sau ~2 token (chung ~0% bài). Chèn
  ở 25% / 50% / 75% bài: lật 7% / 11% / 12% vs placebo 7% / 7% / 0% → gần như hết tác dụng khi trò đã bắt đầu. Gợi ý trong prompt: lật 29%.
  → Nguyên nhân tác động ở LÚC LẬP KẾ HOẠCH (đầu bài); cặp phản thực "chung phần lớn bài, chỉ khác ở bước then chốt" gần như không tồn
  tại với bước của Luna. Mẫu nhỏ (27 bài/điểm cắt). Lưu `experiments/rounds/R05/diag_causal_masked/`.
- **2026-11-28 (trưa)** — Mốc cùng seed `grpo_s3`: @512 TB 19,0 / DEV 24,3, cắt 65,2% (AIME 3,3, AMC 18,8, MATH 44,9, Minerva 15,6, Olym 12,3);
  @2048 TB 32,6 / DEV 39,0. cfpair_reg − grpo CÙNG SEED @512: TB +4,0 / +3,9 / **−0,5** (TB +2,5 ± 1,5 sai số chuẩn); DEV +4,8 / +5,0 / +2,6
  (TB **+4,1 ± 0,8**, cả 3 seed dương). Seed 3 thua TB do AIME/AMC (0,0 / 11,9 vs 3,3 / 18,8; bộ 30/40 câu) — trên 3 bộ lớn vẫn hơn.
  @2048: TB −0,7, DEV −0,4 (ngang). VM c đã dừng. Lưu `experiments/rounds/R03/grpo_s3/`.
- **2026-11-28 (chiều)** — `diag_causal --mask 0` (gợi ý GIỮ số = do(sự kiện), vẫn bỏ bước chứa đáp án và sau nó): **D1 QUA** — 36/120 câu
  khó có bộ đủ thật (BH; placebo 1/120), 87% trong số đó (câu ≥3 bước) chỉ cần 1–2 sự kiện. NHƯNG sự kiện đủ một mình 79% là sự kiện CUỐI
  trước đáp án (còn ~16% y* phía sau) → gần với "cho kết quả trung gian sát đáp án". Tách: câu khó, sự kiện cuối một mình 20,9%; một sự kiện
  sớm 9,4% ≈ placebo 7,2%; MỌI sự kiện trừ cái cuối 18,0%. D2: 0 cạnh qua BH (hiệu ứng TB +6,5 vs placebo +0,8 điểm %). **D4 QUA** lần 2
  (quan sát +0,27 vs can thiệp +0,09; Spearman −0,04, trần 0,42). **D3 QUA** (14 câu, 75 bài sai): chèn sự kiện ở 0% / 25% / 50% / 75% bài lật
  31% / 30% / 22% / 27% vs placebo 1% / 1% / 4% / 3%; cặp lật chung 25% / 50% / 76% bài; chèn rỗng tái tạo 100%; chỉ ≤5% bài lật nhắc tới
  "note". → Với SỰ KIỆN (có số), cặp phản thực tối thiểu tồn tại và sửa được giữa bài; với NƯỚC ĐI che số thì không. Lưu
  `experiments/rounds/R05/diag_causal_unmasked/`.
- **2026-11-28 (chiều)** — Khung mới (người dùng duyệt "thử đi"): SFT+RL = GỘP DỮ LIỆU — rollout của trò là dữ liệu thực nghiệm (không lệch,
  rỗng ở nhóm cả-sai), lời giải thầy là dữ liệu quan sát có biến gây nhiễu ẩn (hiểu biết riêng của thầy). Phương pháp "neo vào thực nghiệm"
  (Kallus–Puli–Shalit 2018): học hàm lệch của tín hiệu thầy d_t = log p(token | ngữ cảnh có y*) − log p(token | đề) trên nhóm lưng chừng,
  dùng tín hiệu đã hiệu chỉnh ở nhóm cả-sai. Viết `code/beat/diag_fusion.py`: T1 nhóm lưng chừng chừa ra (AUC, cos gradient vs GRPO);
  T2 chuyển sang câu cả-sai ở 8 mẫu nhưng giải được với 24 mẫu thêm (tín hiệu chỉ thấy 8 mẫu sai, mốc = GRPO của 32 mẫu). Đối chứng: tín
  hiệu thô, một hệ số, và "không thầy" (chỉ prior theo loại token/vị trí học từ RL). Chạy trên VM d từ 13:11.
- **2026-11-28 (chiều)** — `diag_fusion` kết quả (640 câu; 415 lưng chừng, 220 cả-sai, 93 cả-sai có bài đúng khi thêm 24 mẫu; T2 dùng 40–60 câu):
  **T1 QUA** (cos gradient với GRPO: hiệu chỉnh 0,192 vs thô 0,130 vs không-thầy 0,117; AUC 0,707 / 0,611 / 0,698). **T2 TRƯỢT** (câu cả-sai,
  chỉ nhìn 8 bài sai, so GRPO 32 bài: thô **0,150**, hiệu chỉnh 0,071, không-thầy 0,026; mốc GRPO nửa–nửa 0,026; AUC 32 bài: thô 0,674 >
  hiệu chỉnh 0,646 > không-thầy 0,639). → Phần lệch học ở câu lưng chừng KHÔNG chuyển sang câu khó: phần "prior theo loại token" học từ RL
  (hệ số chặn) làm hỏng; còn tín hiệu thầy thô thì chuyển được. Hệ số khớp: độ dốc theo d ở token CHỮ ≈ 0 (0,001) vs 0,04–0,08 ở token
  toán/số/đáp án → tín hiệu thầy trên chữ (giọng văn) không mang thông tin về đúng/sai, đúng như đo 27/11. Biến thể "chỉ độ dốc, bỏ hệ số
  chặn" là giả thuyết SAU khi xem dữ liệu (chưa chạy). Lưu `experiments/rounds/R05/diag_fusion/`.
- **2026-11-28 (chiều)** — Tìm nguyên nhân T2 trượt (`diag_fusion --diag 1`, CÙNG seed 71 → chỉ để chẩn đoán, không tính là bằng chứng):
  cos với GRPO-32 trên 8 bài sai: chỉ hệ số chặn (prior theo loại token học từ RL) **0,026** ≈ không-thầy; chỉ độ dốc 0,141; thô 0,122;
  hiệu chỉnh đủ 0,079 → thủ phạm là PHẦN PRIOR HỌC TỪ RL, không chuyển sang câu khó. Token chữ: AUC theo loại 0,48 (lưng chừng) / 0,51 (khó)
  = không mang thông tin, nhưng có |d| lớn nhất (0,144 vs 0,025–0,07) và chiếm 30% token → chữ là NHIỄU lấn át tín hiệu thầy. Bỏ chữ: cos
  T2 0,166, chỉ token toán + đáp án 0,175 (mốc nhiễu GRPO nửa–nửa 0,088); T1 bỏ chữ 0,201 vs thô 0,126. Token đáp án AUC 0,81 ở câu khó (thầy
  biết đáp án → gần với chấm đúng/sai, cần tách khi kiểm thật). Phân phối d giữa câu lưng chừng và câu khó gần như giống nhau → không phải
  dịch chuyển phân phối. Đây là giả thuyết SAU dữ liệu, phải kiểm lại trên seed mới. VM d đã tắt. Dữ liệu từng mẫu:
  `experiments/rounds/R05/diag_fusion_diag/samples.pkl`.
- **2026-11-28 (chiều)** — Phân tích offline `samples.pkl` (seed 71, khám phá sau dữ liệu; script `experiments/rounds/R05/diag_fusion_diag/
  fusion_analysis.py`). Môi trường = độ khó (lưng chừng 1–2 / 3–5 / 6–7 đúng trên 8; câu khó 32 mẫu). AUC trong câu của tín hiệu thầy:
  CHỮ 0,40 / 0,55 / 0,45 / 0,51 → ĐỔI CHIỀU giữa các môi trường (không bất biến; ở 1–2/8 còn ngược, CI loại 0,5); token toán + số + khoảng
  trắng (bỏ chữ, bỏ đáp án) 0,58 / 0,67 / 0,60 / 0,63 → ổn định, luôn > 0,5; token ĐÁP ÁN 0,73 / 0,70 / 0,52 / 0,81 (thầy biết đáp án ≈ máy
  chấm mềm, không phải kỹ năng). Độ dài gây nhiễu: tương quan (log độ dài, d trung bình) trong câu +0,49…+0,73 (do lấy trung bình); bỏ độ
  dài vẫn còn 0,50 / 0,63 / 0,60 / 0,64. Thông tin của thầy dồn ở 1/3 CUỐI bài (0,73 ở câu khó) — 1/3 đầu (kế hoạch) chỉ 0,47–0,61.
  SỬA giải thích trước: prior học từ RL KHÔNG mất khả năng xếp hạng ở câu khó (AUC 0,64 / 0,73 / 0,70 / 0,63, khá bất biến); nó trượt T2 vì
  trên 8 bài đều sai nó chỉ đẩy một hướng CHUNG cho mọi câu (bớt chữ, thêm ký hiệu), không mang thông tin riêng của câu. Giá trị của thầy
  nằm ở thông tin RIÊNG của từng câu.
- **2026-11-28 (chiều)** — Placebo seed 3 `cfpair_plc_s3`: @512 TB 19,5 / DEV 27,1, cắt 58,4%; @2048 TB 31,2 / DEV 37,9. Tổng hợp 3 seed (TB / DEV):
  @512 grpo 17,9 / 24,0 · cfpair_reg 20,4 / 28,1 · placebo 19,7 / 27,5 → reg − placebo ghép seed +0,7 TB / +0,6 DEV (DEV +1,5 / +0,6 / −0,2;
  trong nhiễu). @2048 grpo 32,3 / 39,3 · reg 31,6 / 38,9 · placebo 30,5 / 37,6 → reg − placebo +1,1 TB / +1,3 DEV (DEV +2,4 / +0,7 / +0,9, cả 3
  seed dương). Chốt: lợi thế @512 của cfpair ≈ "có cặp tương phản trên y*" (viết gọn), không phải đổi đúng giá trị; đổi đúng giá trị giữ độ
  đúng ở @2048 mà placebo làm mất. VM a đã dừng. Lưu `experiments/rounds/R03/cfpair_plc_s3/`.
- **2026-11-28 (chiều)** — **GRPO phạt độ dài `grpo_len_s1`** (lpen 0,5, mốc 550 token, chỉ phạt bài ĐÚNG dài): @512 TB **21,4** / DEV **28,9**, cắt
  54,4% (AIME 1,7, AMC 18,8, MATH 51,5, Minerva 19,1, Olym 15,9); @2048 TB 31,7 / DEV 39,1. Cùng seed 1: grpo 17,6 / 23,8 · cfpair_reg 21,6 / 28,6
  @512; @2048 grpo 32,4 / 39,9 · reg 30,9 / 39,5. → Một phạt độ dài đơn giản đạt NGANG cfpair_reg ở @512 và giữ độ đúng @2048 (≥ reg). Kết
  luận (1 seed): lợi thế @512 của cfpair là VIẾT GỌN, và viết gọn đạt được không cần y* / cặp phản thực. cfpair_reg không còn là phương pháp
  thắng có ý nghĩa so với mốc "GRPO + phạt độ dài". Cần seed 2–3 của grpo_len để chốt.
- **2026-11-28 (chiều)** — `diag_iv` (seed 73; 900 trạng thái trong bài sai, 733 từ câu cả-sai, 349 câu; 8 đoạn × 3 nhánh): bước đầu: P(xác lập
  sự kiện kế tiếp còn thiếu) 6,5% → 11,6% khi đoạn viết có y* (FS +5,1 điểm %, CI 4,2–6,1); placebo −0,3 (CI chứa 0) → biến công cụ HỢP LỆ
  nhưng dưới ngưỡng đặt trước 0,10 (I1 trượt). Vùng chồng lấn 13% < 20% (I2 trượt). **Dạng rút gọn: chỉ MỘT đoạn ~96 token viết với y* rồi
  bỏ y* đi → tỉ lệ đúng 4,5% → 9,4% (+4,9, CI 4,0–5,7); câu cả-sai 2,7% → 7,5% (~2,8×); placebo +0,0.** LATE 0,95 (CI 0,76–1,20) — gần 1,
  phi lý → điều kiện LOẠI TRỪ không đứng: y* trong một đoạn tác động lên kết quả qua nhiều thứ khác ngoài sự kiện F (cách đặt bài, các bước
  khác; đoạn lộ đáp án 10,3% vs 8,0%). I3 "qua" nhưng không tin được. Tương quan thô trong nhánh Z=0: +0,04. → Định nghĩa treatment theo một
  sự kiện số là quá hẹp; tác dụng thật của thầy nằm ở NỘI DUNG CẢ ĐOẠN. Lưu `experiments/rounds/R05/diag_iv/` (khi tải về).
- **2026-11-28 (tối)** — `diag_mediator` (seed 74; 900 trạng thái, 16.200 đoạn; nhánh đúng z0 5,1% / z1 9,8% / placebo 4,9%): bộ chấm từ trạng thái
  ẩn của trò (PCA-256, hai lớp) có phạt "mômen cửa trước" (dư số trung bình bằng nhau giữa các nhánh). **M1 QUA**: λ = 3000 giải thích 69% tác
  dụng của thầy (ERM thường đã 48%), AUC gộp 0,581 vs ERM 0,585. **M2 TRƯỢT**: trên đoạn TRÒ TỰ VIẾT ở câu cả-sai, AUC trong trạng thái 0,42
  (ERM 0,43) — < 0,5, chỉ 39 trạng thái có cả đúng lẫn sai → không có bộ chấm nào xếp hạng được đoạn của trò ở chỗ cần. Mọi bộ chấm yếu
  (AUC gộp ~0,58). Đối chứng (AUC tính lại có xử lý hoà hạng): F 0,514, độ dài 0,46. → Biểu diễn "mang tác dụng của thầy" tồn tại khi trộn
  đoạn có thầy, nhưng KHÔNG chuyển sang chấm đoạn của chính trò ở câu khó. Cả 4 VM đã tắt. Lưu `experiments/rounds/R05/diag_mediator/`.
