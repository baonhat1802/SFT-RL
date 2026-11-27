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
- **VM đang dùng (từ 2026-11-26): `phuquy-a100`** trong project NHÓM `<team-gcp-project>`, zone **us-central1-b**, a2-highgpu-1g (1× A100
  40GB, 12 vCPU, 83 GB RAM), ổ 200 GB, nhãn owner=phuquy. ~3,67 USD/giờ khi chạy — **tắt khi xong** (đã tắt 26/11 sau kiểm trước R02-1).
  Vào: `gcloud compute ssh phuquy-a100 --zone us-central1-b --project <team-gcp-project>`. Trong project còn VM `hoang` (A100 80GB) của
  bạn cùng nhóm — không đụng.
- Trên VM: `~/miniconda3/envs/capstone` (vllm 0.15.1, trl 0.22.2, transformers 4.57.6, torch 2.9.1 cu128), repo `~/capstone` (bản
  `git archive HEAD` + `code/data/{tracks/curated, bridge/eval_math, bridge_curated_teachers, bridge_curated_teachers_fixed}`), HF `~/hf`.
  Dựng lại: `.work/vm/setup_team_vm.sh` (+ `install.sh`), ~4 phút; gói code+dữ liệu 18 MB + 9 MB.
- VM cũ `capstone-a100-v2` (project cá nhân `<personal-gcp-project>`) không dùng được: **billing của project cá nhân đã tắt** (26/11), kể cả
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
