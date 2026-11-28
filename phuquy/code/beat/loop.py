"""A small GRPO loop with hooks for the R01 ideas (research/rounds/R01/ideas.md). Every R01 run, including the GRPO-only
control (--idea grpo, or any idea with --opt novel=0), runs through THIS loop so they share one framework.

    PYTHONPATH=code python -m beat.loop --idea grpo --seed 1 --out runs/r01/grpo_s1
    PYTHONPATH=code python -m beat.loop --idea guide_tokw --opt tokw=0 --seed 1 --out runs/r01/guide_s1

Generation: vLLM (LoRA adapter re-saved each rollout and served as a new LoRARequest). Training: transformers + peft LoRA
(r32 alpha64, all linear layers), bf16, gradient checkpointing. GRPO as in code/train/common.py: 8 completions per prompt,
advantage = reward - group mean (no std scaling), PPO clip 0.2/0.28, no KL, token-mean loss over each optimizer mini-batch,
2 optimizer steps per rollout, temperature 1.0, answer budget 512 tokens, lr 1e-5 constant.

Ideas (opts are --opt key=value; every idea has novel=0 -> plain GRPO):
  grpo          plain GRPO
  guide_tokw    hint=1: for prompts whose 8 completions all fail, sample k extra completions with y* in the prompt and add them
                        to the group; their tokens are trained under the NO-hint prompt with LUFFY's shaping p/(p+gamma)
                tokw=1: per-token advantage A*w_t, w_t = clip(exp(sign(A)*(logp_T - logp_S)), lo, hi) normalised to mean 1 per
                        completion, teacher T = same model with y* in the prompt (RLSD: reward sets the sign, teacher the size)
  fade          per-prompt hint level p_x in {0,.25,.5,.75}: that share of y* (tokens) is put at the start of the answer as a
                prefix (not trained); all-fail -> p_x up one level, >=2 correct -> down one level
  compress_sd   on correct completions longer than their group median, add on-policy distillation towards the teacher with
                y* (per-token reward -(logp_S - logp_T), coefficient sd_coef)
  opsd          GRPO + (1-lam)*on-policy self-distillation on all-fail groups (or all, allfail_only=0); lam=1 -> GRPO.
  potential     A_t = A + eta * (V(s_k) - V(s_{k-1})) for tokens of segment k, V = exp(log p(answer | prompt, prefix,
                "Therefore, the final answer is \\boxed{")) under the current model; up to n_bound newline cuts per completion
  metaopsd      MAML: theta' = theta - alpha*grad L_SD(theta; S); loss grad g = g_Q(theta') - alpha * H_S g_Q (finite
                differences); theta <- theta' then Adam step with g. S = all-fail groups (query=same: S = the mixed groups
                themselves), Q = groups with non-zero advantage. hessian=0 -> first-order; meta=0 -> opsd with lam.
  cfpair        R03-1: GRPO + lam * [NLL(y*) - NLL(y~*)] (token means), y~* = y* with 1-2 DERIVED values (pert=numbers) or
                +/- signs (pert=operators) changed and every other character kept, so the writing style of y* cancels and only
                "the correct value beats a wrong one in the same context" remains. Pre-screen: runs/diag_causal (+0.10/+0.11 vs
                GRPO +0.07/+0.09). One y~* per question per rollout; novel=0 or lam=0 -> plain GRPO.
                form=linear collapsed (cfpair_s1, the margin grew without bound). form=dpo bounds it: per pair
                h = [mean logp(y*) - ref] - [mean logp(y~*) - ref] (ref = frozen initial model, adapter disabled),
                loss lam * mean softplus(-beta*h)/beta: at h=0 the gradient is lam/2 x the linear one (lam=0.2 matches
                the pre-screened linear lam=0.1), and it vanishes once the policy prefers y* over y~* by ~3/beta nat/token
                more than the initial model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HINT_TEMPLATE = (
    "{question}\n\n"
    "Here is a short reference solution written by an expert. It may skip steps.\n"
    "<reference>\n{reference}\n</reference>\n\n"
    "Now write your own complete solution to the problem. Explain every step clearly, filling in any steps the "
    "reference skips. Do not mention the reference solution; write as if you solved it yourself."
    "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)
# Teacher context used only for SCORING (self-distillation, token weights): concise, unlike HINT_TEMPLATE (which asks for
# a complete solution and is kept for GENERATING hinted completions). With HINT_TEMPLATE as the scoring context the teacher
# prefers verbose continuations and pulls the student towards longer answers, against the 512-token budget (2026-11-21).
TEACHER_TEMPLATE = (
    "{question}\n\n"
    "Here is a reference solution:\n"
    "<reference>\n{reference}\n</reference>\n\n"
    "Solve the problem concisely: go straight to the solution, keep the steps that matter, do not pad. "
    "Do not mention the reference solution."
    "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)
DEFAULT_OPTS = {
    "grpo": {"lpen": 0.0, "ltgt": 550},
    "guide_tokw": {"hint": 1, "tokw": 1, "k_hint": 4, "gamma": 0.1, "w_lo": 0.5, "w_hi": 2.0},
    "fade": {"adapt": 1, "fixed_level": 0.5},
    "compress_sd": {"sd_coef": 0.1, "longonly": 1},
    "opsd": {"lam": 0.5, "allfail_only": 1},
    # MAML-style (R01-1): inner step = one SGD step of on-policy self-distillation on support S (all-fail groups), outer =
    # GRPO loss on query Q (other groups) at the adapted point, second-order term by finite differences (DARTS).
    # R01-6 potential-based shaping (Ng et al. 1999): V(prefix) = P(model writes the correct answer if it stops here),
    # per-segment advantage V(s_k) - V(s_{k-1}) added to the GRPO advantage, segments cut at newlines.
    "potential": {"eta": 1.0, "n_bound": 4},
    "cfpair": {"lam": 0.1, "pert": "numbers", "form": "linear", "beta": 10.0, "neg": 1, "pair": "cf", "inv": 0.0},
    "metaopsd": {"alpha": 1e-3, "inner_norm": 0.1, "hess_clip": 1.0, "keep_inner": 1, "hessian": 1, "query": "other", "meta": 1, "lam": 0.5, "max_s": 32, "eps": 0.01},
}


MATH_RE = None


def perturb(ref, kind, question, rng):
    """Minimal same-style counterfactual of y* (identical to beat/diag_cos.py --causal-sweep)."""
    import re
    global MATH_RE
    if MATH_RE is None:
        MATH_RE = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|\$(.*?)\$", re.S)
    bx = ref.rfind("\\boxed"); bx = len(ref) if bx < 0 else bx
    spans = [(m.start(), min(m.end(), bx)) for m in MATH_RE.finditer(ref) if m.start() < bx]
    given = set(re.findall(r"\d+", question))
    if kind == "numbers":
        cand = [(st + m.start(), st + m.end()) for st, en in spans for m in re.finditer(r"\d+", ref[st:en])
                if m.group() not in given]
    elif kind == "prose":      # placebo: swap 1-2 prose words (outside math, before \\boxed) — validity unchanged
        inside = lambda c: any(st <= c < en for st, en in spans)
        cand = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]{4,}", ref[:bx]) if not inside(m.start())]
        words = sorted({ref[u:v] for u, v in cand})
        if len(words) < 2:
            return ref
    else:
        cand = [(st + m.start(), st + m.end()) for st, en in spans for m in re.finditer(r"[+-]", ref[st:en])]
    if not cand:
        return ref
    pick = sorted(rng.sample(cand, min(len(cand), rng.choice([1, 2]))))
    out, last = [], 0
    for u, v in pick:
        t = ref[u:v]
        if kind == "numbers":
            n = int(t); new = str(n + rng.choice([1, 2, 3]) if n < 3 else n + rng.choice([-2, -1, 1, 2, 3]))
        elif kind == "prose":
            new = rng.choice([w for w in words if w != t])
        else:
            new = "-" if t == "+" else "+"
        out += [ref[last:u], new]; last = v
    out.append(ref[last:]); return "".join(out)


def parse_opts(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, choices=list(DEFAULT_OPTS))
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-rollouts", type=int, default=None, help="smoke tests")
    ap.add_argument("--prompts-per-rollout", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=8, help="samples per micro-batch when --tok-budget 0")
    ap.add_argument("--tok-budget", type=int, default=4096,
                    help="micro-batches packed by length under this many padded tokens (beat/fastlp.py); 0 = old path")
    ap.add_argument("--vllm-sleep", type=int, default=1, help="free vLLM's GPU memory between generations")
    ap.add_argument("--vllm-mem", type=float, default=0.7, help="0.7 assumes --vllm-sleep 1 (vLLM idle during training)")
    ap.add_argument("--reward-workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-repo", default=None, help="private HF model repo: checkpoint pushed every --save-every rollouts "
                                                       "and pulled at start (molab sessions end after 12 h)")
    ap.add_argument("--save-every", type=int, default=5)
    a = ap.parse_args()
    opts = {**DEFAULT_OPTS[a.idea], **parse_opts(a.opt)}
    novel = int(opts.get("novel", 1))

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.common import BASE_MODEL, GRPO, LORA, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    log_f = open(out / "log.jsonl", "a")
    (out / "config.json").write_text(json.dumps({"args": vars(a), "opts": opts}, indent=2))

    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    eos_id = tok.convert_tokens_to_ids(tok.eos_token)

    df = load_track("curated")
    if a.limit:
        df = df.head(a.limit)
    rows = []
    for r in df.itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        hint_msgs = [msgs[0], {"role": "user", "content": HINT_TEMPLATE.format(
            question=r.extra_info["question"], reference=r.target[0]["content"])}]
        hp = tok.apply_chat_template(hint_msgs, tokenize=False, add_generation_prompt=True)
        t_msgs = [msgs[0], {"role": "user", "content": TEACHER_TEMPLATE.format(
            question=r.extra_info["question"], reference=r.target[0]["content"])}]
        tp = tok.apply_chat_template(t_msgs, tokenize=False, add_generation_prompt=True)
        rows.append({"p_ids": tok(p, add_special_tokens=False)["input_ids"],
                     "hp_ids": tok(hp, add_special_tokens=False)["input_ids"],
                     "tp_ids": tok(tp, add_special_tokens=False)["input_ids"],
                     "ref_ids": tok(r.target[0]["content"], add_special_tokens=False)["input_ids"],
                     "gt": r.reward_model["ground_truth"],
                     "ref_txt": r.target[0]["content"], "q_txt": r.extra_info["question"],
                     "ans_ids": tok(str(r.extra_info["answer"]) + "}", add_special_tokens=False)["input_ids"]})
    rows = [x for x in rows if len(x["p_ids"]) <= 1024 and len(x["hp_ids"]) <= 2048]
    level = np.zeros(len(rows))          # fade: hint level per prompt
    SUFFIX_IDS = tok("\nTherefore, the final answer is \\boxed{", add_special_tokens=False)["input_ids"]
    NL_IDS = None
    if a.idea == "potential":
        NL_IDS = {i for i in range(len(tok)) if "\n" in tok.decode([i])}

    # ---- models
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = {**LORA, "r": a.lora_r, "lora_alpha": a.lora_alpha}
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lcfg))
    torch.manual_seed(a.seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    llm_kw = dict(model=BASE_MODEL, enable_lora=True, max_lora_rank=max(64, a.lora_r), max_loras=1, max_cpu_loras=2,
                  gpu_memory_utilization=a.vllm_mem, max_model_len=2048 + a.max_completion, seed=a.seed,
                  enable_prefix_caching=True, enable_sleep_mode=bool(a.vllm_sleep))
    if os.environ.get("UNSLOTH_VLLM_NO_FLASHINFER", "0") != "0":      # molab Blackwell: FlashInfer JIT fails
        llm_kw["attention_backend"] = "FLASH_ATTN"
    try:
        llm = LLM(**llm_kw)
    except TypeError:
        llm_kw.pop("attention_backend", None)
        llm = LLM(**llm_kw)
    sp = SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion,
                        stop_token_ids=stop_ids, n=a.num_gen)
    sp1 = SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion,
                         stop_token_ids=stop_ids, n=int(opts.get("k_hint", 4)))

    from beat import fastlp

    def mbs(items):
        """Micro-batches of samples: packed by length under --tok-budget, or fixed --micro-batch chunks (old path)."""
        if a.tok_budget > 0:
            lens = [len(x["ctx"]) + len(x["comp"]) for x in items]
            return [[items[i] for i in b] for b in fastlp.token_batches(lens, a.tok_budget)]
        return [items[i:i + a.micro_batch] for i in range(0, len(items), a.micro_batch)]

    def logps_of(ctx_ids_list, comp_ids_list, grad: bool):
        """Per-token log-prob of each completion given its context. Returns list of 1-D tensors (float32)."""
        if a.tok_budget > 0:      # no-grad passes keep no activations: twice the budget
            return fastlp.logps(model, tok.pad_token_id, ctx_ids_list, comp_ids_list, grad,
                                a.tok_budget if grad else 2 * a.tok_budget)
        # Left padding, so every completion ends at the last position; only the last K = max completion length + 1
        # logits are materialised (logits_to_keep), not the logits of the (possibly long, hinted) context.
        res = []
        for i in range(0, len(ctx_ids_list), a.micro_batch):
            ctxs, comps = ctx_ids_list[i:i + a.micro_batch], comp_ids_list[i:i + a.micro_batch]
            seqs = [c + d for c, d in zip(ctxs, comps)]
            L = max(len(s) for s in seqs)
            K = max(len(d) for d in comps) + 1
            ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, L - len(s):] = torch.tensor(s)
                att[j, L - len(s):] = 1
            ids, att = ids.cuda(), att.cuda()
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            with torch.set_grad_enabled(grad):
                logits = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=K).logits
                # logits[:, k] predicts the token at position L - K + 1 + k
                for j, d in enumerate(comps):
                    n = len(d)
                    lg = logits[j, K - 1 - n:K - 1].float()
                    tgt = ids[j, L - n:]
                    res.append(torch.log_softmax(lg, -1).gather(-1, tgt[:, None]).squeeze(-1))
            del logits
        return res

    params = [p for p in model.parameters() if p.requires_grad]
    eps_lo, eps_hi = GRPO["epsilon"], GRPO["epsilon_high"]

    def accumulate(smps, kind):
        """Backward the token-mean loss of `smps` (kind 'grpo' = PPO-clip surrogate with s['adv'], 'sd' = on-policy
        reverse-KL surrogate sum_t sg(logp_S - logp_T) * logp_S). Gradients accumulate into p.grad."""
        ntok = sum(len(x["comp"]) for x in smps)
        tot = 0.0
        for mb in mbs(smps):
            new = logps_of([x["ctx"] for x in mb], [x["comp"] for x in mb], grad=True)
            loss = 0.0
            for x, lp in zip(mb, new):
                if kind == "grpo":
                    adv = x["adv"].to(lp.device)
                    ratio = (lp - x["old"]).exp()
                    loss = loss - torch.minimum(ratio * adv, ratio.clamp(1 - eps_lo, 1 + eps_hi) * adv).sum()
                else:
                    loss = loss + ((lp.detach() - x["lpT"].to(lp.device)) * lp).sum()
            (loss / ntok).backward()
            tot += float(loss) / ntok
        return tot

    def grads():
        return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]

    def zero():
        for p in params:
            p.grad = None

    def set_params(vals):
        with torch.no_grad():
            for p, v in zip(params, vals):
                p.copy_(v)

    def dot(u, v):
        return sum(float((x.float() * y.float()).sum()) for x, y in zip(u, v))

    def meta_update(support, query):
        """Two MAML steps per rollout (halves of S and Q), Adam on the meta-gradient; the inner step is kept."""
        # inner step size: inner_norm > 0 -> ||theta' - theta|| = inner_norm (alpha = inner_norm / ||g_S||, treated as a
        # constant in the meta-gradient); one Adam step of the LoRA (~60M params, lr 1e-5) has norm ~0.08. Smoke with a fixed
        # alpha = 1e-3 gave hess_rel ~1e-3, i.e. a no-op inner step (2026-11-21).
        st = {"cos_gQ_gS": [], "hess_rel": [], "loss_q": [], "loss_s": [], "alpha_eff": [], "gS_norm": []}
        hs = [support[: (len(support) + 1) // 2], support[(len(support) + 1) // 2:]]
        hq = [query[: (len(query) + 1) // 2], query[(len(query) + 1) // 2:]]
        for S_h, Q_h in zip(hs, hq):
            if not Q_h:
                continue
            theta0 = [p.detach().clone() for p in params]
            if S_h:
                zero(); st["loss_s"].append(accumulate(S_h, "sd")); gS = grads()
                gs_norm = math.sqrt(dot(gS, gS)) + 1e-12
                alpha = opts["inner_norm"] / gs_norm if opts["inner_norm"] > 0 else opts["alpha"]
                st["alpha_eff"].append(alpha); st["gS_norm"].append(gs_norm)
                set_params([t - alpha * g for t, g in zip(theta0, gS)])
            theta1 = [p.detach().clone() for p in params]
            zero(); st["loss_q"].append(accumulate(Q_h, "grpo")); gQ = grads()
            meta = gQ
            if S_h:
                nq, ns = math.sqrt(dot(gQ, gQ)) + 1e-12, math.sqrt(dot(gS, gS)) + 1e-12
                st["cos_gQ_gS"].append(dot(gQ, gS) / (nq * ns))
                if opts["hessian"]:
                    e = opts["eps"] / nq
                    set_params([t + e * g for t, g in zip(theta0, gQ)]); zero(); accumulate(S_h, "sd"); gp = grads()
                    set_params([t - e * g for t, g in zip(theta0, gQ)]); zero(); accumulate(S_h, "sd"); gm = grads()
                    hvp = [(x - y) / (2 * e) for x, y in zip(gp, gm)]
                    rel = alpha * math.sqrt(dot(hvp, hvp)) / nq
                    # Smoke @2048 (2026-11-21): rel reached 60 when few query groups had non-zero advantage -> the update
                    # was finite-difference noise. Cap the second-order term at hess_clip * ||g_Q||.
                    scale = min(1.0, opts["hess_clip"] / rel) if opts["hess_clip"] > 0 and rel > 0 else 1.0
                    meta = [q - scale * alpha * h for q, h in zip(gQ, hvp)]
                    st["hess_rel"].append(rel)
                    st.setdefault("hess_scale", []).append(scale)
            # keep_inner=1: the inner (teacher) step stays in the model. metaopsd_s1 (2026-11-21) collapsed with it: cos(g_Q, g_S)
            # stayed at -0.15..-0.36, so every kept teacher step hurt reward on other questions (reward 0.27 -> 0.02 by rollout 6).
            # keep_inner=0 is standard MAML: the inner step is only a look-ahead; the model moves by the meta-gradient alone.
            set_params(theta1 if opts["keep_inner"] else theta0)
            for p, g in zip(params, meta):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            zero()
        return {k: (float(np.mean(v)) if v else None) for k, v in st.items()}

    n_prompts = len(rows)
    steps_per_epoch = n_prompts // a.prompts_per_rollout
    total = a.max_rollouts or int(math.ceil(a.epochs * steps_per_epoch))
    order = []
    rng = np.random.default_rng(a.seed)
    while len(order) < total * a.prompts_per_rollout:
        order.extend(rng.permutation(n_prompts).tolist())
    # ---- checkpoint / resume (local out/ckpt, mirrored to the HF Hub when --hub-repo)
    ckpt = out / "ckpt"
    api = None
    if a.hub_repo:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()
        api.create_repo(a.hub_repo, private=True, exist_ok=True)
        if not (ckpt / "meta.json").exists():
            try:
                snapshot_download(a.hub_repo, allow_patterns=["ckpt/*"], local_dir=str(out))
            except Exception as e:                       # empty repo
                print(f"no hub checkpoint ({type(e).__name__})", flush=True)
    start = 0
    if (ckpt / "meta.json").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        meta = json.loads((ckpt / "meta.json").read_text())
        set_peft_model_state_dict(model, load_file(str(ckpt / "adapter_model.safetensors")))
        opt.load_state_dict(torch.load(ckpt / "optim.pt", map_location="cuda"))
        if (ckpt / "levels.npy").exists():
            level = np.load(ckpt / "levels.npy")
        if (ckpt / "log.jsonl").exists() and not (out / "log.jsonl").stat().st_size:
            (out / "log.jsonl").write_text((ckpt / "log.jsonl").read_text())
        start = meta["step"] + 1
        print(f"RESUME from rollout {start} ({ckpt})", flush=True)

    def save_ckpt(step):
        ckpt.mkdir(exist_ok=True)
        model.save_pretrained(str(ckpt))
        torch.save(opt.state_dict(), ckpt / "optim.pt")
        np.save(ckpt / "levels.npy", level)
        log_f.flush()
        (ckpt / "log.jsonl").write_text((out / "log.jsonl").read_text())
        (ckpt / "meta.json").write_text(json.dumps({"step": step, "total": total}))
        if api is not None:
            try:
                api.upload_folder(repo_id=a.hub_repo, folder_path=str(ckpt), path_in_repo="ckpt",
                                  commit_message=f"rollout {step}", ignore_patterns=["README.md"])
            except Exception as e:
                print(f"hub upload failed: {e}", flush=True)

    print(f"START idea={a.idea} opts={opts} prompts={n_prompts} rollouts={total} from={start}", flush=True)

    for step in range(start, total):
        if step > start and (step - start) % a.save_every == 0:
            save_ckpt(step - 1)
        t0 = time.time()
        if a.vllm_sleep and step > start:
            torch.cuda.empty_cache(); llm.wake_up()
        adir = out / "adapter" / f"r{step % 2}"
        model.save_pretrained(str(adir))
        lreq = LoRARequest(f"a{step}", step + 1, str(adir))
        idx = order[step * a.prompts_per_rollout:(step + 1) * a.prompts_per_rollout]

        # ---- build contexts (fade: prefix of y* inside the answer)
        ctx, prefix = [], []
        for i in idx:
            pre = []
            if a.idea == "fade" and novel:
                lv = level[i] if opts["adapt"] else 0.0
                pre = rows[i]["ref_ids"][:int(lv * len(rows[i]["ref_ids"]))]
            ctx.append(rows[i]["p_ids"] + pre)
            prefix.append(pre)
        outs = llm.generate([{"prompt_token_ids": c} for c in ctx], sp, lora_request=lreq, use_tqdm=False)
        samples = []          # dicts: g (group), ctx, comp, text, hinted
        for g, (i, o) in enumerate(zip(idx, outs)):
            for c in o.outputs:
                comp = list(c.token_ids)
                if c.finish_reason == "stop" and (not comp or comp[-1] not in stop_ids):
                    comp.append(eos_id)
                pre_text = tok.decode(prefix[g]) if prefix[g] else ""
                samples.append({"g": g, "i": i, "ctx": ctx[g], "comp": comp, "text": pre_text + c.text,
                                "trunc": c.finish_reason == "length", "hinted": False})
        t_gen = time.time() - t0
        rew = score_all([s["text"] for s in samples], [rows[s["i"]]["gt"] for s in samples], a.reward_workers)
        for s, r in zip(samples, rew):
            s["r"] = r
        G = a.num_gen
        group_r = np.array(rew).reshape(len(idx), G)
        allfail = group_r.sum(1) == 0
        m = {"step": step, "reward": float(group_r.mean()), "allfail": float(allfail.mean()),
             "allcorrect": float((group_r.sum(1) == G).mean()),
             "len": float(np.mean([len(s["comp"]) for s in samples])),
             "trunc": float(np.mean([s["trunc"] for s in samples]))}

        # ---- fade: update levels
        if a.idea == "fade" and novel:
            for g, i in enumerate(idx):
                if opts["adapt"]:
                    if allfail[g]:
                        level[i] = min(0.75, level[i] + 0.25)
                    elif group_r[g].sum() >= 2:
                        level[i] = max(0.0, level[i] - 0.25)
                elif allfail[g]:
                    level[i] = opts["fixed_level"]
            m["fade_level_mean"] = float(level.mean())
            m["fade_level_nonzero"] = float((level > 0).mean())

        # ---- guide: hinted completions for all-fail groups (trained under the no-hint prompt)
        if a.idea == "guide_tokw" and novel and opts["hint"] and allfail.any():
            fail_g = [g for g in range(len(idx)) if allfail[g]]
            hout = llm.generate([{"prompt_token_ids": rows[idx[g]]["hp_ids"]} for g in fail_g], sp1,
                                lora_request=lreq, use_tqdm=False)
            extra = []
            for g, o in zip(fail_g, hout):
                for c in o.outputs:
                    comp = list(c.token_ids)
                    if c.finish_reason == "stop" and (not comp or comp[-1] not in stop_ids):
                        comp.append(eos_id)
                    extra.append({"g": g, "i": idx[g], "ctx": rows[idx[g]]["p_ids"], "comp": comp, "text": c.text,
                                  "trunc": c.finish_reason == "length", "hinted": True})
            er = score_all([s["text"] for s in extra], [rows[s["i"]]["gt"] for s in extra], a.reward_workers)
            for s, r in zip(extra, er):
                s["r"] = r
            m["hint_correct"] = float(np.mean(er)) if er else 0.0
            samples += extra

        if a.vllm_sleep:          # generation for this rollout is done: give the GPU memory to training
            llm.sleep(level=1)

        # ---- control: GRPO with a length penalty on CORRECT samples only (length-matched baseline for cfpair_reg);
        # r = 1 - lpen * max(0, len - ltgt) / ltgt (floored at 0); wrong samples stay 0; logged reward above stays correctness
        if a.idea == "grpo" and opts.get("lpen", 0) > 0:
            for s in samples:
                if s["r"] == 1:
                    s["r"] = max(0.0, 1.0 - opts["lpen"] * max(0, len(s["comp"]) - opts["ltgt"]) / opts["ltgt"])
            m["shaped_reward"] = float(np.mean([s["r"] for s in samples]))

        # ---- advantages per group (over everything in the group, hinted included)
        by_g = {}
        for s in samples:
            by_g.setdefault(s["g"], []).append(s["r"])
        for s in samples:
            s["A"] = s["r"] - float(np.mean(by_g[s["g"]]))

        # ---- potential-based shaping: value of prefixes = P(correct answer | prefix)
        if novel and a.idea == "potential":
            model.eval()
            q_ctx, q_ans, owners = [], [], []
            for si, smp in enumerate(samples):
                comp = smp["comp"]
                nls = [j for j, t in enumerate(comp[:-1]) if t in NL_IDS]
                if len(nls) > opts["n_bound"]:
                    pick = np.linspace(0, len(nls) - 1, int(opts["n_bound"])).round().astype(int)
                    nls = [nls[k] for k in sorted(set(pick.tolist()))]
                cuts = [0] + [j + 1 for j in nls]
                smp["cuts"] = cuts
                for c in cuts:
                    q_ctx.append(smp["ctx"] + comp[:c] + SUFFIX_IDS)
                    q_ans.append(rows[smp["i"]]["ans_ids"])
                    owners.append(si)
            vals = logps_of(q_ctx, q_ans, grad=False)
            vs = {}
            for si, lp in zip(owners, vals):
                vs.setdefault(si, []).append(math.exp(float(lp.sum())))
            dsum = []
            for si, smp in enumerate(samples):
                V, cuts = vs[si], smp["cuts"]
                dense = torch.zeros(len(smp["comp"]))
                for k in range(1, len(cuts)):                    # segment [cuts[k-1], cuts[k]) gets V_k - V_{k-1}
                    dense[cuts[k - 1]:cuts[k]] = V[k] - V[k - 1]
                smp["dense"] = dense
                dsum.append(float(dense.abs().sum()))
            m["V0_mean"] = float(np.mean([vs[i][0] for i in vs]))
            m["Vlast_mean"] = float(np.mean([vs[i][-1] for i in vs]))
            m["dense_abs_sum"] = float(np.mean(dsum))

        # ---- teacher log-probs (same model with y* in the prompt) where needed
        need_T = []
        if novel and a.idea == "guide_tokw" and opts["tokw"]:
            need_T = [s for s in samples if s["A"] != 0 and not s["hinted"]]
        if novel and a.idea == "compress_sd":
            for g in range(len(idx)):
                ss = [s for s in samples if s["g"] == g and s["r"] == 1.0]
                if not ss:
                    continue
                med = float(np.median([len(s["comp"]) for s in samples if s["g"] == g]))
                for s in ss:
                    if not opts["longonly"] or len(s["comp"]) > med:
                        s["sd"] = True
                        need_T.append(s)
        if novel and a.idea == "metaopsd":
            mixed = [g for g in range(len(idx)) if 0 < group_r[g].sum() < G]
            s_groups = mixed if opts["query"] == "same" else [g for g in range(len(idx)) if allfail[g]]
            s_groups = set(s_groups[:int(opts["max_s"])])
            for smp in samples:
                if opts.get("sd_skip_trunc", 1) and smp["trunc"]:
                    continue
                if smp["g"] in s_groups:
                    smp["sd" if not opts["meta"] else "S"] = True
                    need_T.append(smp)
            m["n_support_groups"] = len(s_groups)
        if novel and a.idea == "opsd" and opts["lam"] < 1:
            for s in samples:
                if opts.get("sd_skip_trunc", 1) and s["trunc"]:
                    continue
                if not opts["allfail_only"] or allfail[s["g"]]:
                    s["sd"] = True
                    need_T.append(s)
        t1 = time.time()
        if need_T:
            model.eval()
            tkey = "hp_ids" if opts.get("teacher", "concise") == "full" else "tp_ids"
            tl = logps_of([rows[s["i"]][tkey] for s in need_T], [s["comp"] for s in need_T], grad=False)
            sl = logps_of([s["ctx"] for s in need_T], [s["comp"] for s in need_T], grad=False)
            for s, t, l in zip(need_T, tl, sl):
                s["lpT"], s["lpS0"] = t, l
            m["n_teacher"] = len(need_T)
            m["kl_ST"] = float(np.mean([(s["lpS0"] - s["lpT"]).mean().item() for s in need_T]))

        # ---- per-token advantage tensors
        train = [s for s in samples if s["A"] != 0 or s.get("sd")
                 or ("dense" in s and float(s["dense"].abs().sum()) > 0)]
        for s in train:
            n = len(s["comp"])
            adv = torch.full((n,), float(s["A"]))
            if "lpT" in s and a.idea == "guide_tokw":
                d = (s["lpT"] - s["lpS0"]).cpu() * (1.0 if s["A"] > 0 else -1.0)
                w = d.exp().clamp(opts["w_lo"], opts["w_hi"])
                adv = adv * (w / w.mean())
            if "dense" in s:
                adv = adv + opts["eta"] * s["dense"]
            s["adv"] = adv
        m["n_train"] = len(train)
        if not train:
            m["time"] = time.time() - t0
            log_f.write(json.dumps(m) + "\n"); log_f.flush()
            continue

        # ---- old log-probs (policy before this rollout's updates)
        model.eval()
        olds = logps_of([s["ctx"] for s in train], [s["comp"] for s in train], grad=False)
        for s, o in zip(train, olds):
            s["old"] = o.detach()
        t_old = time.time() - t1

        if novel and a.idea == "metaopsd" and opts["meta"]:
            model.train()
            support = [x for x in samples if x.get("S")]
            query = [x for x in train if x["A"] != 0]
            random.shuffle(support); random.shuffle(query)
            mstats = meta_update(support, query)
            m.update(mstats)
            m.update({"t_gen": round(t_gen, 1), "t_old": round(t_old, 1), "time": round(time.time() - t0, 1)})
            log_f.write(json.dumps(m) + "\n"); log_f.flush()
            print("STEP " + json.dumps(m), flush=True)
            continue

        # ---- cfpair: same-style counterfactual pairs (y*, y~*) for this rollout's questions, split over the two halves
        cf_halves = [[], []]
        if novel and a.idea == "cfpair" and opts["lam"] > 0:
            prng = random.Random(a.seed * 100003 + step)
            prng_inv = random.Random(a.seed * 100003 + step + 7)
            pairs = []
            for g_, i in enumerate(idx):
                if opts["pair"] == "natural":      # ablation: DPO on natural pairs, y* vs one of the student's own wrong solutions
                    wr = [s_ for s_ in samples if s_["g"] == g_ and s_["r"] == 0 and not s_["trunc"]]
                    if wr:
                        c_ids = list(prng.choice(wr)["comp"])
                        if not c_ids or c_ids[-1] != eos_id:
                            c_ids.append(eos_id)
                        pairs.append((rows[i]["p_ids"], rows[i]["ref_ids"] + [eos_id], c_ids, None))
                    continue
                kind = "prose" if opts["pair"] == "placebo" else opts["pert"]   # placebo: style-only counterfactual
                t = perturb(rows[i]["ref_txt"], kind, rows[i]["q_txt"], prng)
                if t != rows[i]["ref_txt"]:
                    pl = None
                    if opts["inv"] > 0:            # counterfactual invariance: a style-only twin of y* (own RNG, value pairs unchanged)
                        tp = perturb(rows[i]["ref_txt"], "prose", rows[i]["q_txt"], prng_inv)
                        pl = tok(tp, add_special_tokens=False)["input_ids"] + [eos_id] if tp != rows[i]["ref_txt"] else None
                    pairs.append((rows[i]["p_ids"], rows[i]["ref_ids"] + [eos_id],
                                  tok(t, add_special_tokens=False)["input_ids"] + [eos_id], pl))
            prng.shuffle(pairs)
            if opts["form"] == "dpo":                 # reference token-mean log-probs, frozen initial model
                model.eval()
                with torch.no_grad(), model.disable_adapter():
                    ry = logps_of([q[0] for q in pairs], [q[1] for q in pairs], grad=False)
                    rc = logps_of([q[0] for q in pairs], [q[2] for q in pairs], grad=False)
                    wpl = [k for k, q in enumerate(pairs) if q[3] is not None]
                    rpl_l = logps_of([pairs[k][0] for k in wpl], [pairs[k][3] for k in wpl], grad=False) if wpl else []
                rpl = {k: float(v.mean()) for k, v in zip(wpl, rpl_l)}
                pairs = [(q[0], q[1], q[2], float(a_.mean()), float(b_.mean()), q[3], rpl.get(k))
                         for k, (q, a_, b_) in enumerate(zip(pairs, ry, rc))]
            else:
                pairs = [(q[0], q[1], q[2]) for q in pairs]
            cf_halves = [pairs[: (len(pairs) + 1) // 2], pairs[(len(pairs) + 1) // 2:]]
            m["cf_n"] = len(pairs)
        cf_marg, cf_loss, cf_h, cf_w, cf_hpl = [], [], [], [], []

        def cf_backward(pairs):
            """lam * [token-mean NLL(y*) - token-mean NLL(y~*)], backward in chunks of 8; logs the margin before the step."""
            if not pairs:
                return
            if opts["form"] == "dpo":
                P, bt, tot = len(pairs), float(opts["beta"]), 0.0
                for k0 in range(0, P, 8):
                    ch = pairs[k0:k0 + 8]
                    lpy = logps_of([q_[0] for q_ in ch], [q_[1] for q_ in ch], grad=True)
                    # ablation neg=0: keep only the bounded pull toward y* (no counterfactual push), same form and weight
                    lpc = logps_of([q_[0] for q_ in ch], [q_[2] for q_ in ch], grad=True) if opts["neg"] else [None] * len(ch)
                    # counterfactual invariance: keep the policy INDIFFERENT to style-only twins (squared margin vs the frozen model)
                    ipl = [k for k, q_ in enumerate(ch) if opts["inv"] > 0 and q_[5] is not None]
                    lpl = dict(zip(ipl, logps_of([ch[k][0] for k in ipl], [ch[k][5] for k in ipl], grad=True))) if ipl else {}
                    loss_c = 0.0; loss_i = 0.0
                    for k, (q_, y_, c_) in enumerate(zip(ch, lpy, lpc)):
                        h = (y_.mean() - q_[3]) - ((c_.mean() - q_[4]) if c_ is not None else 0.0)
                        loss_c = loss_c + torch.nn.functional.softplus(-bt * h) / bt
                        cf_h.append(float(h.detach())); cf_w.append(float(torch.sigmoid(-bt * h.detach())))
                        cf_marg.append(float(y_.detach().mean() - (c_.detach().mean() if c_ is not None else q_[3])))
                        if k in lpl:
                            hp = (y_.mean() - q_[3]) - (lpl[k].mean() - q_[6])
                            loss_i = loss_i + hp ** 2
                            cf_hpl.append(float(hp.detach().abs()))
                    ((opts["lam"] * loss_c + opts["inv"] * loss_i) / P).backward()
                    tot += float(loss_c) * opts["lam"] / P + float(loss_i) * opts["inv"] / P
                cf_loss.append(tot)
                return
            nty = sum(len(y) for _, y, _ in pairs); ntc = sum(len(c) for _, _, c in pairs)
            tot = 0.0
            for k0 in range(0, len(pairs), 8):
                ch = pairs[k0:k0 + 8]
                lpy = logps_of([p_ for p_, _, _ in ch], [y for _, y, _ in ch], grad=True)
                ly = -opts["lam"] * sum(l.sum() for l in lpy) / nty
                ly.backward()
                lpc = logps_of([p_ for p_, _, _ in ch], [c for _, _, c in ch], grad=True)
                lc = opts["lam"] * sum(l.sum() for l in lpc) / ntc
                lc.backward()
                tot += float(ly) + float(lc)
                cf_marg.extend(float(y_.detach().mean() - c_.detach().mean()) for y_, c_ in zip(lpy, lpc))
            cf_loss.append(tot)

        # ---- 2 optimizer steps over halves
        model.train()
        random.shuffle(train)
        halves = [train[: (len(train) + 1) // 2], train[(len(train) + 1) // 2:]]
        losses = []
        for hi_, half in enumerate(halves):
            if not half:
                continue
            ntok = sum(len(s["comp"]) for s in half)
            opt.zero_grad(set_to_none=True)
            for mb in mbs(half):
                new = logps_of([s["ctx"] for s in mb], [s["comp"] for s in mb], grad=True)
                loss = 0.0
                for s, lp in zip(mb, new):
                    adv = s["adv"].to(lp.device)
                    if s["hinted"]:                                   # LUFFY shaping, no ratio/clip
                        p = lp.exp()
                        term = -(adv * (p / (p + opts["gamma"]))).sum()
                    else:
                        ratio = (lp - s["old"]).exp()
                        un = ratio * adv
                        cl = ratio.clamp(1 - eps_lo, 1 + eps_hi) * adv
                        term = -torch.minimum(un, cl).sum() if bool((adv != 0).any()) else lp.sum() * 0
                    if s.get("sd"):                                   # on-policy distillation to the y* teacher
                        r_t = -(s["lpS0"] - s["lpT"]).to(lp.device)
                        coef = opts.get("sd_coef", 1 - opts.get("lam", 1.0))
                        term = term - coef * (r_t * lp).sum()
                    loss = loss + term
                (loss / ntok).backward()
                losses.append(float(loss) / ntok)
            cf_backward(cf_halves[hi_])
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            m.setdefault("gnorm", []).append(round(float(gn), 4))
            opt.step()
        if cf_marg:
            m["cf_margin"] = float(np.mean(cf_marg))        # mean logp/token: y* minus y~* (> 0: prefers the correct value)
            m["cf_loss"] = float(np.sum(cf_loss))
        if cf_h:
            m["cf_h"] = float(np.mean(cf_h)); m["cf_w"] = float(np.mean(cf_w))      # margin vs initial model; mean sigmoid weight
        if cf_hpl:
            m["cf_hpl"] = float(np.mean(cf_hpl))     # |margin| y* vs its style-only twin (invariance term drives it to 0)
        m.update({"loss": float(np.mean(losses)), "t_gen": round(t_gen, 1), "t_old": round(t_old, 1),
                  "time": round(time.time() - t0, 1)})
        log_f.write(json.dumps(m) + "\n"); log_f.flush()
        print("STEP " + json.dumps(m), flush=True)

    save_ckpt(total - 1)
    model.save_pretrained(str(out / "final_adapter"))
    tok.save_pretrained(str(out / "final_adapter"))
    if api is not None:
        api.upload_folder(repo_id=a.hub_repo, folder_path=str(out / "final_adapter"), path_in_repo="final_adapter",
                          commit_message="final adapter", ignore_patterns=["README.md"])
    if a.idea == "fade":
        np.save(out / "fade_levels.npy", level)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
