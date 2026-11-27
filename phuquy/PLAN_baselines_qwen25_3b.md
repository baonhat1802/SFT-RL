# Plan: chạy lại các baseline SFT + RL với Qwen2.5-3B, hai bộ dữ liệu, bằng Unsloth trên molab

Lập 2026-11-12, thay plan cũ (Qwen2.5-Math-1.5B, setup LUFFY đầy đủ: mỗi lần RL ước 15–30 giờ, quá nặng).
Sửa cùng ngày: train bằng **Unsloth + TRL** thay cho verl; **LoRA** thay cho full fine-tuning.
Chưa chạy gì. Mọi con số thời gian là **ước lượng**, đo lại ở Bước 1 rồi sửa plan.

## 0. Cấu hình đã chốt

| | Giá trị | Lý do |
|---|---|---|
| Model | **Qwen2.5-3B** (base) | model chính của BRIDGE; ngữ cảnh 32K sẵn; không dùng dòng Qwen2.5-Math vì RL trên dòng đó tăng điểm cả khi phần thưởng ngẫu nhiên (*Spurious Rewards*, 2025), khó tin kết quả |
| Framework | **Unsloth** (`FastLanguageModel` + `get_peft_model`) + **TRL** (`SFTTrainer`, `GRPOTrainer`), sinh mẫu bằng vLLM (`fast_inference=True`) | một tiến trình trên một GPU, không cần Ray (máy molab chỉ có 4 CPU); vLLM và trainer dùng chung trọng số trong bộ nhớ; phương pháp của nhóm sau này viết trên TRL dễ hơn trên verl |
| Kiểu train | **LoRA cho mọi baseline** và cho phương pháp của nhóm (cấu hình ở mục 4) | LoRA + `fast_inference` là đường được Unsloth hỗ trợ tốt nhất; checkpoint chỉ vài trăm MB nên lưu và chạy tiếp qua nhiều phiên molab nhanh; ít bộ nhớ hơn |

**Hệ quả của việc dùng LoRA, phải ghi trong báo cáo:**

- Các paper so sánh (LUFFY, SRFT, ReLIFT, SFT-then-RL, BRIDGE) đều full fine-tune. Số của ta **không so trực tiếp** được
  với bảng của họ; phép so là giữa các baseline của nhóm, cùng điều kiện LoRA.
- LoRA thường học ít hơn và quên ít hơn full fine-tuning (*LoRA Learns Less and Forgets Less*, 2024). Ảnh hưởng mạnh nhất
  ở SFT trên nhiều dữ liệu (bộ A); với RL, LoRA thường theo kịp full fine-tuning (*LoRA Without Regret*, Thinking Machines,
  2025). Nghĩa là khoảng cách giữa SFT và RL trong bảng có thể khác với khoảng cách khi full fine-tune.
- BRIDGE gốc dùng LoRA làm "thầy" đặt trên một model full fine-tune. Trên nền LoRA, phương pháp kiểu BRIDGE sẽ cần **hai
  adapter** (một cho trò, một cho thầy); tính trước khi thiết kế phương pháp của nhóm.
| Bộ train A `luffy8k` | 8.000 câu lấy ngẫu nhiên (seed 2026) từ Openr1-Math-46k-8192, chỉ giữ câu có lời giải R1 ≤ 4.096 token **và** được `reward.py` chấm đúng | lời giải dài kiểu R1 có sẵn, giống loại dữ liệu BRIDGE dùng |
| Bộ train B `curated` | 8.519 câu MATH mức 3–5 của nhóm (`BaoNhat/bridge-math-curated-teachers`), lời giải GPT-6 Luna **bản đã sửa định dạng** | dữ liệu riêng của nhóm; lời giải ngắn (trung vị 115 token) |
| SFT và RL | dùng **cùng tập câu hỏi** trong mỗi bộ (như BRIDGE) | |
| Reward khi train | `reward.py` của nhóm (`compute_score`), **cho cả hai bộ** | chấm đúng 100% lời giải mẫu của cả hai bộ; nghiêm hơn Math-Verify ở đáp án nhiều giá trị |
| Độ dài trả lời khi train | tối đa 4.096 token; prompt tối đa 1.024 | như BRIDGE với model 3B |
| Đánh giá | 9 bộ của setup LUFFY (`all.parquet`), nhiệt độ 0,6, tối đa 8.192 token, avg@32 cho AIME24/AIME25/AMC | bộ đánh giá đầy đủ nhất đang có |
| Leo bench | `dev.parquet`: MATH-500 + Minerva + OlympiadBench, bỏ 39 câu gần trùng với một trong hai bộ train → **1.408 câu** | AIME, AMC và 3 bộ ngoài phân phối chỉ chạy ở lần đánh giá cuối |

Dữ liệu đã dựng: `code/data/tracks/{luffy8k,curated}/train.parquet` (`code/scripts/make_tracks.py`), một schema chung:
`prompt`, `reward_model`, `target` (lời giải mẫu cho LUFFY/SRFT/ReLIFT), `messages` (cho SFT), `extra_info`.
TRL đọc trực tiếp được: `prompt` là danh sách tin nhắn chat, `messages` là dạng hội thoại cho `SFTTrainer`.

| | Bộ A `luffy8k` | Bộ B `curated` |
|---|---|---|
| Số câu | 8.000 | 8.519 |
| Lời giải mẫu: trung vị / tối đa | 2.624 / 4.096 token | 115 / 987 token |
| Trùng với bộ đánh giá (≥ 50% cụm 13 từ) | 0 | 32 câu (MATH-500 19, MMLU-Pro 8, OlympiadBench 5); phần lớn là bài na ná khác số liệu, vài câu giống hệt |
| Prompt khi train **và** khi đánh giá | kiểu LUFFY: system prompt "Your task is to follow a systematic, thorough reasoning process…", câu trả lời mở bằng `<think>` | kiểu SimpleRL: "You are a helpful assistant" + "Please reason step by step, and put your final answer within \boxed{}" |

**Đã cân nhắc và không chọn (2026-11-12):**

- Trần 6.144 hoặc 8.192 token cho bộ A: giữ được 83% hoặc 100% số bài thay vì 56%, bớt lệch về bài dễ, nhưng mỗi lần
  RL tốn khoảng 1,5 hoặc 2 lần.
- Dùng cả 24.585 bài đạt điều kiện thay vì 8.000: RL tốn như cũ, SFT lâu khoảng 3 lần.

Giữ 4.096 token và 8.000 bài: rẻ nhất, và 4.096 trùng với cấu hình BRIDGE cho Qwen2.5-3B. Cái giá cần ghi trong báo cáo:
bộ A lệch về bài có lời giải ngắn, thường là bài dễ hơn.

Mỗi bộ dùng prompt riêng của nó cho cả train lẫn đánh giá, vì lời giải bộ B không có `<think>`. Model gốc (chưa train)
được đánh giá với **cả hai** prompt.

## 1. Baseline

Mỗi baseline có train chạy **riêng cho từng bộ** (GRPO cũng vậy, vì hai bộ khác câu hỏi).

| # | Baseline | Làm gì | Trên Unsloth + TRL |
|---|---|---|---|
| B0 | Qwen2.5-3B gốc | chỉ đánh giá, với cả hai prompt | — |
| B1 | SFT | 3 epoch trên lời giải mẫu | `SFTTrainer`, chỉ tính loss trên câu trả lời |
| B2 | GRPO | RL từ model gốc, không SFT | `GRPOTrainer` |
| B3 | SFT→RL | GRPO tiếp từ B1 | `GRPOTrainer`: gộp adapter B1 vào model gốc (`merge_and_unload`), rồi gắn adapter LoRA mới cho RL |
| B4 | LUFFY | trộn lời giải mẫu vào nhóm mẫu GRPO | **phải tự viết**: lớp con của `GRPOTrainer` |
| B5 | ReLIFT | xen kẽ SFT và RL | **phải tự viết** |
| B6 | SRFT | loss SFT + loss RL trong cùng batch | **phải tự viết** |

B1–B3 có sẵn trong TRL. B4–B6 chỉ có bản cài trên verl (repo `alek6kun/sft_then_rl` @ `e0f3573`, thư mục
`verl/verl/{mix_src,relift_src,srft_src}`), nên phải **chuyển sang TRL**, dùng code verl đó làm bản tham chiếu.

**Mọi baseline và phương pháp của nhóm phải chạy trên cùng framework.** Không trộn số verl với số Unsloth: hai framework
khác nhau ở cách gộp loss, cắt tỉ lệ xác suất, xử lý câu bị cắt độ dài. Paper SFT-then-RL cho thấy chính những chi tiết
kiểu này đã làm sai lệch cả một nhánh paper.

**Thứ tự ưu tiên:** B0 → bộ A: B1, B2, B3 → bộ B: B1, B2, B3 → B4 → B5, B6.
B1–B3 là ba mốc paper SFT-then-RL (NeurIPS 2026) cho là mạnh nhất; mọi phương pháp mới phải thắng B3.

## 2. Bước 1 (làm trước, khoảng một ngày): dựng môi trường, kiểm pipeline, đo tốc độ

Không chạy baseline nào trước khi xong bước này.

### 2.1 Máy molab

1 × RTX PRO 6000 Blackwell 96 GB, 4 CPU, 32 GB RAM. Phiên tối đa 12 giờ; rảnh 90 phút thì tắt. Bộ nhớ lâu dài có giới
hạn. Notebook public nhưng không ai tìm thấy nếu không có link.

Hệ quả:

- Mọi lần chạy phải **lưu checkpoint và chạy tiếp** được qua nhiều phiên.
- Checkpoint đẩy lên Hugging Face Hub (repo private), không giữ trên molab.
- Token HF, W&B, GitHub để trong secret của molab, không ghi vào code.
- Quy định molab có câu "notebook chạy việc không tương tác có thể bị dừng không báo": thêm lý do để lưu checkpoint thường.

### 2.2 Cài đặt (`notebooks/00_env_smoke.py`)

1. Clone repo này (private: cần token GitHub trong secret).
2. Cài `unsloth` bản mới nhất, `vllm` bản CUDA 12.8 (`cu128`), `triton>=3.3.1`. Theo tài liệu Unsloth, Blackwell (kể cả
   RTX PRO 6000) cần đúng các điều kiện này. Thêm `math-verify==0.8.0`, `sympy==1.14.0`,
   `latex2sympy2_extended==1.10.2`, `antlr4-python3-runtime==4.13.2` (theo `reward_requirements.txt`), `wandb`,
   `huggingface_hub`.
3. Ghi lại phiên bản chính xác của `unsloth`, `trl`, `transformers`, `vllm`, `torch` vào `experiments/env.txt`. Mọi
   baseline chạy trên **đúng bộ phiên bản đó**.
4. Tải dữ liệu từ Kaggle dataset `nguyenlamphuquy/gstar-capstone-sft-rl-math` (public).

Qua bước khi:

- `FastLanguageModel.from_pretrained("Qwen/Qwen2.5-3B", load_in_4bit=False, fast_inference=True, max_lora_rank=64)`
  rồi `get_peft_model(...)` theo cấu hình LoRA ở mục 4 chạy. **Không** nạp 4-bit: model gốc giữ bf16 cho mọi baseline.
- Sinh được 8 câu trả lời bằng vLLM, kể cả khi đã gắn adapter.
- `GRPOTrainer` chạy 2 bước và `SFTTrainer` chạy 20 bước không lỗi.
- Dừng giữa chừng rồi chạy tiếp từ checkpoint thành công (`resume_from_checkpoint`).
- Gộp một adapter SFT vào model (cho B3) rồi gắn adapter mới, sinh mẫu vẫn đúng.

Unsloth xác nhận hỗ trợ RTX PRO 6000; LoRA + GRPO + `fast_inference` là tổ hợp chuẩn của Unsloth. Nếu `fast_inference`
vẫn không chạy trên Blackwell: để TRL tự quản vLLM (`use_vllm=True`, `vllm_mode="colocate"`); nếu vẫn không được, quay
về verl (verl cũng hỗ trợ LoRA).

### 2.3 Hai bản model

| Tên | Từ | Chat template | Dùng cho |
|---|---|---|---|
| `Qwen2.5-3B-think` | `Qwen/Qwen2.5-3B` | template kiểu LUFFY lấy từ `sft_then_rl/models/README.md` (system prompt + `<think>`) | bộ A |
| `Qwen2.5-3B` | nguyên bản | ChatML mặc định; prompt SimpleRL đã nằm sẵn trong dữ liệu | bộ B |

### 2.4 Kiểm pipeline

**Điểm neo đánh giá.** Chạy Qwen2.5-3B gốc, **giải mã greedy**, trả lời tối đa 4.096 token, prompt SimpleRL, trên
MATH-500. BRIDGE báo **32,4** cho đúng cấu hình này. Lệch ≤ 3 điểm thì qua; lệch hơn thì sửa đánh giá trước khi train gì.
Đây là điểm neo duy nhất đúng model; các điểm neo LUFFY (1.5B, 7B) không áp dụng cho Qwen2.5-3B.

**Kiểm hai lỗi mà paper SFT-then-RL chỉ ra**, vì đổi framework thì phải kiểm lại:

1. *Cộng dồn gradient:* một bước SFT với batch 8 phải cho **cùng gradient** (sai khác dưới 1e-3 tương đối) với 4 lần
   cộng dồn batch 2, trên cùng 8 mẫu có độ dài khác nhau. Lệch là loss đang chia sai mẫu số.
2. *Optimizer:* không dùng optimizer offload sang CPU. Nếu Unsloth tự bật, tắt đi. Với LoRA thì trạng thái optimizer nhỏ,
   không có lý do để offload.

**Reward:** hàm reward gói `reward.py` phải cho 1,0 trên 50 lời giải mẫu và 0,0 trên 50 lời giải sai do đổi đáp án.

### 2.5 Đo tốc độ, rồi sửa mục 4

| Chạy thử | Đo |
|---|---|
| 10 bước GRPO bộ A, 10 bước GRPO bộ B | giây/bước; độ dài trả lời trung bình; phần trăm thời gian dành cho sinh mẫu |
| 100 bước SFT bộ A | token/giây |
| lưu + đẩy một checkpoint lên HF | phút; chọn tần suất lưu |
| bật 2 notebook GPU cùng lúc trên một tài khoản | có chạy song song được không |

Lưu ý khi đọc số: với RL, phần lớn thời gian là **sinh mẫu bằng vLLM**, Unsloth không làm phần này nhanh hơn nhiều. Phần
Unsloth tăng tốc rõ là bước cập nhật trọng số và bộ nhớ. Nên đừng kỳ vọng RL nhanh gấp nhiều lần so với ước lượng ở mục 4.

## 3. Đánh giá (`notebooks/10_eval.py`)

- Script: `code/evaluation/run.py`. Chấm **y như** `LUFFY/eval_scripts/generate_vllm.py` (repo `ElliottYan/LUFFY` @
  `01f4ae0`): với prompt `<think>` chỉ chấm phần sau `</think>`; Math-Verify, kèm bộ chấm OAT của LUFFY. Khác LUFFY ở
  một chỗ: prompt dựng **giống hệt lúc train** cho từng bộ. Template `simplerl` của LUFFY viết `within\boxed{{}}`
  (thiếu dấu cách, thừa ngoặc), lệch với prompt train của bộ B.
- Chấm bằng Math-Verify, báo thêm bộ chấm OAT.
- Báo hai bản: bộ chuẩn, và bộ đã bỏ 40 câu gần trùng (`code/data/tracks/eval_exclude_union.tsv`).
- Trong lúc train: chỉ chạy `dev.parquet` mỗi 25 bước RL (vài phút). Bộ đầy đủ: chỉ ở checkpoint cuối.
- Bảng kết quả: 6 bộ toán (AIME24, AIME25, AMC, MATH-500, Minerva, OlympiadBench) + 3 bộ ngoài phân phối (ARC-c,
  GPQA Diamond, MMLU-Pro), một hàng cho mỗi (baseline, bộ train).

## 4. Cấu hình train

Giữ siêu tham số của SFT-then-RL (bản verl), dịch sang tham số TRL. Số bước theo cỡ dữ liệu, độ dài theo BRIDGE.
Learning rate nhân 10 so với bản full fine-tuning, theo khuyến nghị của *LoRA Without Regret* (lr tối ưu của LoRA gấp
khoảng 10 lần full fine-tuning, cả SFT lẫn RL).

### LoRA (chung cho mọi baseline)

| Tham số | Giá trị | Lý do |
|---|---|---|
| `r` | 64 | đủ chỗ cho SFT bộ A (khoảng 21 triệu token mỗi epoch); với RL, rank thấp đã đủ nên 64 là dư |
| `lora_alpha` | 64 | `alpha = r`, mặc định của Unsloth |
| `lora_dropout`, `bias` | 0, `"none"` | Unsloth tối ưu cho cấu hình này |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | gắn vào **mọi** lớp tuyến tính, kể cả MLP; *LoRA Without Regret* cho thấy chỉ gắn vào attention thì kém hẳn |
| `use_gradient_checkpointing` | `"unsloth"` | |
| `random_state` | theo seed của run | |
| Model gốc | bf16, **không** nạp 4-bit | |

Kiểm nhanh trước khi chạy dài: B2 bộ A chạy 30 bước với lr 1e-5. Nếu reward trung bình không tăng, thử thêm 3e-6 và 3e-5
(30 bước mỗi giá trị) và chọn **một** lr dùng cho mọi bản RL. Làm tương tự cho SFT nếu loss của B1 đi ngang.

### SFT (`SFTTrainer`)

| Tham số TRL | Giá trị | Ứng với verl (`verl_sft.sh`) |
|---|---|---|
| `num_train_epochs` | 3 | `trainer.total_epochs=3` |
| `learning_rate`, `lr_scheduler_type`, `warmup_ratio` | **5e-4**, `cosine`, 0,1 | verl full fine-tune dùng 5e-5 (×10 cho LoRA); verl còn có `min_lr_ratio=0.1`: dùng `cosine_with_min_lr` |
| batch hiệu dụng | 64 (`per_device_train_batch_size` × `gradient_accumulation_steps`) | `train_batch_size=64` |
| `max_length` | 5.120 | prompt 1.024 + trả lời 4.096 |
| `completion_only_loss` | `True` | verl chỉ tính loss trên câu trả lời |
| `bf16`, `gradient_checkpointing` | `True`, `"unsloth"` | |

### GRPO (`GRPOTrainer`)

| Tham số TRL | Giá trị | Ứng với verl (`verl_rl.sh`) |
|---|---|---|
| `num_generations` | 8 | `rollout.n=8` |
| số câu mỗi lượt sinh | 128 câu × 8 = 1.024 lời giải (`generation_batch_size=1024`) | `train_batch_size=128` |
| số lần cập nhật mỗi lượt sinh | 2, mỗi lần 512 lời giải (`per_device_train_batch_size` × `gradient_accumulation_steps` = 512; TRL tự suy ra `steps_per_generation` = 1.024 / micro-batch, đếm theo **micro-batch**) | `ppo_mini_batch_size=64` |
| `num_train_epochs` | **1** (chốt 2026-11-12, phương án C): bộ A khoảng 62 lượt sinh = **125 bước cập nhật** TRL, bộ B khoảng 67 lượt sinh = 133 bước | BRIDGE dùng 2 epoch; giảm vì đo được 317 s mỗi lượt sinh. Thuật toán giữ nguyên (2 lần cập nhật mỗi lượt sinh như verl), áp dụng cho mọi baseline |
| `learning_rate` | **1e-5**, hằng số | verl full fine-tune dùng `optim.lr=1e-6` (×10 cho LoRA) |
| `beta` | 0 | `use_kl_loss=false` |
| `epsilon`, `epsilon_high` | 0,2, 0,28 | `clip_ratio_low/high` |
| `scale_rewards` | `False` | `norm_adv_by_std_in_grpo=false` |
| `loss_type` | `"dapo"` (chuẩn hoá theo token) | verl token-mean. Khác một chỗ: TRL chia cho số token của **cả lượt sinh** (1.024 lời giải), verl chia cho số token của **mini-batch** (512), nên gradient TRL nhỏ khoảng một nửa. Adam gần như không nhạy với hệ số tỉ lệ cố định; như nhau cho mọi baseline |
| `temperature` | 1,0 | `rollout.temperature=1.0` |
| `max_prompt_length`, `max_completion_length` | 1.024, 4.096 | |
| `mask_truncated_completions` | `False` | verl không loại câu bị cắt độ dài; ghi rõ nếu đổi |
| `reward_funcs` | một hàm gói `reward.py::compute_score` | reward chung |
| `save_steps` | 10 bước cập nhật (= 5 lượt sinh, khoảng 20–35 phút); SFT mỗi 25 bước (khoảng 7 phút). Mỗi lần đẩy lên HF khoảng 10 giây | molab |
| `generation_kwargs` | `{"stop_token_ids": [<|im_end|> hoặc <|endoftext|>, <|endoftext|>]}` | vLLM chỉ dừng ở token được báo; không có dòng này thì lời giải bộ B chạy tới 4.096 token |

Seed: 1 seed cho mọi bản trước; thêm seed 2, 3 cho B2 và B3 nếu còn giờ.

### B4–B6 (tự viết, chuyển từ verl)

| | Ý chính cần chuyển | File verl tham chiếu |
|---|---|---|
| LUFFY | trong mỗi nhóm 8 lời giải, thay một lời giải bằng lời giải mẫu; lời giải mẫu được tính trọng số bằng hàm uốn `p/(p+0,1)` thay cho tỉ lệ xác suất, không cắt | `mix_src/mix_core_alg.py`, cấu hình `off_policy_reshape="p_div_p_0.1"` trong `verl_luffy_rl.sh` |
| SRFT | cộng loss SFT trên lời giải mẫu vào loss GRPO trong cùng batch, trọng số theo entropy | `srft_src/` |
| ReLIFT | gom các câu mà cả 8 lời giải đều sai, định kỳ SFT trên lời giải mẫu của các câu đó | `relift_src/`, `sft.sft_epochs=1`, `optim.sft.lr=1e-6` (×10 cho LoRA) |

Cả ba chỉ train **một** adapter LoRA, như B2.

Mỗi bản phải qua một **phép thử đối chiếu**: trên cùng một batch tensor cố định, loss và gradient của bản TRL khớp bản
verl (sai khác dưới 1e-4 tương đối). Không chạy dài trước khi khớp.

### Thời gian đo được (molab, 2026-11-12, `experiments/smoke_speed.json`)

| Việc | Bộ A | Bộ B | Ước lượng trước đó |
|---|---|---|---|
| SFT 3 epoch | **1,8 giờ** (17,5 s/bước) | vài phút (chưa đo) | 1–2 giờ |
| Một lần GRPO 2 epoch, từ model gốc | **14,6 giờ** (210 s/bước cập nhật) | **15,0 giờ** (203 s/bước) | A 6–12 giờ, B 3–6 giờ |
| Đánh giá MATH-500, greedy | khoảng 2 phút | | |

Một lượt sinh (1.024 lời giải, trung bình khoảng 650 token với model gốc) mất khoảng 7 phút: khoảng 4,5 phút sinh
và chấm, rồi 2 lần cập nhật khoảng 75 giây mỗi lần. Tốc độ sinh chỉ khoảng 2.500 token/giây, thấp xa so với
khoảng 13.500 token/giây khi vLLM chạy model gốc không LoRA ở bước điểm neo; bộ chấm cũng báo nhiều lần
`Timeout during comparison`. Chưa biết phần nào chiếm thời gian; cần đo trước khi cắt bớt. Sau SFT bộ A, model viết dài
như lời giải R1 (khoảng 2.600 token), nên B3 bộ A sẽ còn chậm hơn.

Đo từng phần (bước `profile`) rồi tăng tốc: vLLM sinh song song 512 thay vì 128, micro-batch cập nhật 8 thay vì 2,
mẩu tính xác suất 8 lời giải. Một lượt sinh từ 410 s xuống **317 s** (sinh 81 s ở 8.455 token/giây; tính xác suất cũ
khoảng 90 s; cập nhật khoảng 83 s; chấm 8 s). Chấm điểm không phải nút thắt. Bỏ được bước tính xác suất cũ nếu chỉ
cập nhật 1 lần mỗi lượt sinh, nhưng như thế đổi thuật toán, nên không làm. **Chốt: RL 1 epoch cho mọi baseline**,
khoảng 5,5 giờ mỗi lần GRPO từ model gốc; B3 bộ A lâu hơn vì model viết dài sau SFT.

Ưu tiên B0–B3 cả hai bộ: khoảng 40–70 giờ GPU. Đủ B0–B6 cả hai bộ: khoảng 90–150 giờ GPU. Một notebook chạy nối tiếp:
1–2 tuần; chia 3–4 notebook song song (nếu tài khoản cho phép, hoặc mỗi thành viên một notebook trên tài khoản riêng):
3–5 ngày. **Không** lập thêm tài khoản để vượt giới hạn (quy định molab tính là lạm dụng).

**Ngưỡng cắt** nếu đo ở 2.5 ra chậm hơn ước lượng 2 lần: giảm RL còn 1 epoch cho **mọi** bản cùng lúc, không giảm riêng
một bản. LoRA chủ yếu giảm bộ nhớ và kích thước checkpoint; thời gian RL vẫn do sinh mẫu quyết định, nên ước lượng trên
không giảm nhiều so với full fine-tuning.

## 5. Code và notebook

Code train nằm trong repo, notebook chỉ gọi vào, để mọi notebook chạy đúng một bản code. Thư mục `code/` trùng tên
module `code` của Python nên không import được như package: chạy với `PYTHONPATH=code`, ví dụ
`PYTHONPATH=code python -m train.grpo --track luffy8k --baseline grpo --seed 1 --hub-repo <user>/<repo>`.

| File | Nội dung | Trạng thái |
|---|---|---|
| `code/train/common.py` | mọi hằng số của mục 4 (LoRA, SFT, GRPO, độ dài), chat template, token dừng mỗi bộ | xong |
| `code/train/luffy_chat_template.jinja` | template kiểu LUFFY, chép nguyên văn từ `sft_then_rl/models/README.md` | xong |
| `code/train/data.py` | dataset SFT (prompt–completion đã render bằng template của bộ) và GRPO (`prompt`, `ground_truth`) | xong |
| `code/train/rewards.py` | hàm reward cho TRL, gói `reward.py::compute_score` của nhóm | xong |
| `code/train/models.py` | nạp model + LoRA qua Unsloth (GPU) hoặc peft (test) | xong |
| `code/train/hub.py` | đẩy checkpoint lên HF sau mỗi lần lưu, giữ 2 bản; tải bản mới nhất để chạy tiếp | xong |
| `code/train/sft.py` | B1; cuối cùng lưu adapter và bản gộp bf16 cho B3 | xong |
| `code/train/grpo.py` | B2, B3 (`--baseline grpo` hoặc `sft_rl --init-model <bản gộp B1>`) | xong |
| `code/train/checks.py` | phép thử cộng dồn gradient, dùng được với model tí hon và model thật | xong |
| `code/evaluation/run.py` | sinh bằng vLLM và chấm (mục 3) | xong |
| `code/tests/` | template, reward, cộng dồn gradient (chỉ chạy khi có GPU) | xong; 10 test không cần model qua trên Mac |
| `code/train/luffy.py`, `srft.py`, `relift.py` | B4–B6 | chưa viết |

| Notebook | Việc | Trạng thái |
|---|---|---|
| `notebooks/00_env_smoke.py` | Bước 1: cài, kiểm, chạy thử, đo (`python -m train.smoke all`) | **xong, mọi bước PASS** (2026-11-12) |
| `notebooks/01_train_baselines.py` | Hàng đợi 14 việc (`python -m train.queue`): B0 đánh giá model gốc cả hai bộ → bộ A: B1, đánh giá, B2, đánh giá, B3, đánh giá → bộ B như vậy. Việc đã xong trên HF được bỏ qua, việc dở chạy tiếp từ checkpoint | viết xong 2026-11-12 |
| `40_mixed.py` | B4, B5, B6 | chưa viết |

Cả hai notebook chỉ kéo code rồi gọi lệnh trong repo; tự chạy khi mở, in tiến độ mỗi phút; không cần upload lại khi code
đổi. Token điền một lần trong cell đầu (bản trong repo để trống).

Mẫu chung cho mọi notebook train:

1. ô cài môi trường (đúng phiên bản trong `experiments/env.txt`);
2. ô tải model và dữ liệu;
3. ô chạy train bằng `subprocess` (`python -m train.<x> ... --hub-repo ...`), ghi log ra file; script tự tìm checkpoint
   mới nhất trên HF và chạy tiếp;
4. ô hiện 50 dòng log cuối, tự làm mới (để phiên không bị coi là rảnh).

## 6. Lưu trữ

- Dữ liệu: Kaggle dataset `nguyenlamphuquy/gstar-capstone-sft-rl-math` (public).
- Model và checkpoint: HF Hub, repo private `<hf-user>/capstone-<track>-<baseline>-seed<s>`; chỉ lưu adapter LoRA và
  trạng thái optimizer của adapter (vài trăm MB), không lưu cả model. Riêng B1 lưu thêm bản đã gộp để B3 nạp.
- Log: W&B project `gstar-capstone`, tên run `<track>_<baseline>_seed<s>`.
- Kết quả đánh giá: JSON trong repo này, `experiments/<track>/<baseline>/eval_<step>.json`.

## 7. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Unsloth GRPO + `fast_inference` không chạy trên Blackwell | lỗi lúc nạp model hoặc lúc sinh | theo thứ tự ở 2.2: tắt `fast_inference` → quay về verl |
| lr ×10 không hợp với LoRA ở model này | reward GRPO đi ngang, hoặc loss SFT không giảm | phép thử lr ở mục 4; chọn một lr chung cho mọi bản |
| SFT bằng LoRA trên bộ A học không đủ | B1 bộ A thấp bất thường so với B2 | ghi lại như một hệ quả của LoRA; không đổi rank riêng cho một bản |
| Bản tự viết B4–B6 sai so với bản gốc | phép thử đối chiếu không khớp | không chạy dài; sửa trước |
| TRL chuẩn hoá loss khác verl | phép thử cộng dồn gradient lệch | sửa trước khi chạy B1 |
| Model 3B đọc nhầm template (bộ A) | câu trả lời không mở `<think>`, không có `\boxed{}` | kiểm 20 mẫu sinh thử ở Bước 1 |
| SFT bộ B (lời giải ngắn) làm hại suy luận | B1 bộ B thấp hơn B0 | đây là một kết quả, ghi lại, không sửa |
| Điểm neo 2.4 không khớp | lệch hơn 3 điểm | dừng, sửa đánh giá trước |
| Hết giờ GPU | — | thứ tự ưu tiên ở mục 1, ngưỡng cắt ở mục 4 |

## 8. Danh sách việc

- [x] Dựng hai bộ train, bộ dev, danh sách câu cần loại (2026-11-12)
- [x] Kaggle dataset có `tracks/` (public)
- [ ] Tạo HF repo private cho model; tạo W&B project
- [x] 2.2 môi trường Unsloth chạy trên molab, ghi `experiments/env.txt` (2026-11-12)
- [x] 2.4 phép thử cộng dồn gradient (fp32, khớp cách chuẩn hoá đúng), reward, token dừng; điểm neo MATH-500: model gốc 58,4 (Math-Verify) / 55,4 (OAT). BRIDGE báo 32,4 với prompt không công bố, chỉ ghi để đối chiếu
- [x] 2.5 đo tốc độ (xem mục 4) → [ ] tăng tốc hoặc cắt còn 1 epoch
- [x] `code/train/` cho B1–B3, `code/evaluation/run.py`, notebook `00_env_smoke.py` (2026-11-12, chưa chạy trên GPU)
- [ ] B0 cả hai prompt (việc 1–2 của `train.queue`)
- [ ] B1–B3 bộ A, rồi bộ B
- [ ] B4–B6: viết, qua phép thử đối chiếu, rồi chạy
- [ ] Bảng kết quả cuối: hai bộ train × các baseline, bộ chuẩn và bộ đã lọc trùng
