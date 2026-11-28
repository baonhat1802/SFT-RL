"""Diagnostic for R01 (no training): is the y*-teacher's self-distillation gradient aligned with the reward gradient?

Both Meta-OPSD runs collapsed with cos(g_Q, g_S) < 0 from rollout 0. This script measures, at the initial policy, the cosine
between the reward gradient on one half of the questions (B) and several guidance gradients on the OTHER half (A), so a
positive value means "a guidance step on A raises reward on unseen questions B" (to first order).

Anchor: cos(g_RL(A), g_RL(B)), the alignment of two GRPO gradients on disjoint questions. A guidance gradient is useful
only if its cosine with g_RL(B) is positive and not far below this anchor.

Guidance configurations (all use the R01 surrogate sum_t sg(logp_S - logp_T) * logp_S, token mean):
  sd_allfail      support = all samples of all-fail groups, teacher = TEACHER_TEMPLATE   (= what Meta-OPSD used)
  sd_mixed_wrong  support = incorrect samples of mixed groups
  sd_correct      support = correct samples of mixed groups
  sd_allfail_ans  support = all-fail, teacher sees only the final answer
  sd_allfail_full support = all-fail, teacher = HINT_TEMPLATE (verbose)
  sd_allfail_pre  support = all-fail, tokens from the first "boxed" token on are masked out (answer-leak test)
  grpo_tokw       GRPO on A with RLSD token weights (reward sets the sign, teacher the size) - the planned next method

    PYTHONPATH=code python -m beat.diag_cos --batches 3 --prompts 128 --max-completion 2048 --out runs/diag_cos
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

ANSWER_TEMPLATE = (
    "{question}\n\n"
    "The final answer is {answer}.\n\n"
    "Solve the problem concisely and arrive at this answer. Do not mention that the answer was given."
    "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--prompts", type=int, default=128, help="prompts per batch (split in halves A and B)")
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--tok-budget", type=int, default=4096, help="beat/fastlp.py packing; 0 = old fixed micro-batch")
    ap.add_argument("--vllm-mem", type=float, default=0.4)
    ap.add_argument("--max-support", type=int, default=160, help="cap on samples per guidance gradient")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter to measure at (default: initial policy)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--adapted-sweep", action="store_true",
                    help="only: teacher = model after a per-question SFT step on y* (y* in the WEIGHTS, not the context), "
                         "vs the in-context teacher")
    ap.add_argument("--adapt-eps", default="0.05,0.1,0.3", help="norms of the per-question SFT step on the LoRA")
    ap.add_argument("--format-sweep", action="store_true",
                    help="only: teacher-context formats x supports -> transfer cosine + teacher-student log-ratio stats")
    ap.add_argument("--tokw-sweep", action="store_true",
                    help="only the RLSD dose-response: anchor + token-weight ranges [1,1] .. [0.5,2] and the inverted sign")
    ap.add_argument("--ystar-split", action="store_true",
                    help="why does SFT on y* oppose the reward? split the y* SFT gradient by token class (results after '=', "
                         "other digits, answer box, plan/prose) + anchor: SFT on the student's own correct solutions")
    ap.add_argument("--causal-sweep", action="store_true",
                    help="R03 pre-screens: counterfactual same-style pairs (y* vs y* with wrong values), invariant (AND-mask) "
                         "gradients across voices (y* / own correct solutions / GRPO), placebo-differenced teacher")
    ap.add_argument("--dr-sweep", action="store_true",
                    help="R04-1: y*-teacher log-ratio as an action-dependent control variate with exact full-vocabulary "
                         "correction (AIPW / doubly robust): E[update] = GRPO gradient; does it reduce GRPO noise?")
    ap.add_argument("--dr-samples", type=int, default=192, help="random samples per half (all groups, A = 0 included)")
    ap.add_argument("--skeleton-sweep", action="store_true",
                    help="causal discovery on y*: delete each step, measure the drop in log-prob of every later step and of the "
                         "answer -> step DAG; skeleton = ancestors of the answer. Transfer cosine of SFT on skeleton vs the rest")
    ap.add_argument("--skel-taus", default="0.1,0.5", help="edge thresholds: mean per-token log-prob drop (nats)")
    ap.add_argument("--model", default=None, help="override the base model (student-size sweep)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat.loop import HINT_TEMPLATE, TEACHER_TEMPLATE
    from train.common import BASE_MODEL, GRPO, LORA, configure_tokenizer
    if a.model:
        BASE_MODEL = a.model
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    eos_id = tok.convert_tokens_to_ids(tok.eos_token)

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        q, ref, ans = r.extra_info["question"], r.target[0]["content"], str(r.extra_info["answer"])
        enc = lambda user: tok(tok.apply_chat_template([msgs[0], {"role": "user", "content": user}], tokenize=False,
                                                       add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        rows.append({"p_ids": tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                                  add_special_tokens=False)["input_ids"],
                     "tp_ids": enc(TEACHER_TEMPLATE.format(question=q, reference=ref)),
                     "hp_ids": enc(HINT_TEMPLATE.format(question=q, reference=ref)),
                     "ap_ids": enc(ANSWER_TEMPLATE.format(question=q, answer=ans)),
                     "ref_ids": tok(ref, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids(tok.eos_token)],
                     # base-model-friendly contexts (2026-11-26): y* as the model's own earlier answer / in the system turn /
                     # as the start of its own answer
                     "mt_ids": tok(tok.apply_chat_template([msgs[0], msgs[1], {"role": "assistant", "content": ref}, msgs[1]],
                                                           tokenize=False, add_generation_prompt=True),
                                   add_special_tokens=False)["input_ids"],
                     "sys_ids": tok(tok.apply_chat_template([{"role": "system", "content": msgs[0]["content"]
                                                              + "\n\nReference solution to the user's problem:\n" + ref}, msgs[1]],
                                                           tokenize=False, add_generation_prompt=True),
                                    add_special_tokens=False)["input_ids"],
                     "pre_ids": tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                                    + ref + "\n\nLet me now write the full solution again, step by step.\n",
                                    add_special_tokens=False)["input_ids"],
                     "ref_txt": ref, "q_txt": q, "sys_txt": msgs[0]["content"],
                     "gt": r.reward_model["ground_truth"]})
    rows = [x for x in rows if len(x["p_ids"]) <= 1024 and max(len(x["hp_ids"]), len(x["mt_ids"]), len(x["pre_ids"])) <= 2048]
    BOXED = {i for i in range(len(tok)) if "boxed" in tok.decode([i])}

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **{**LORA, "r": 32, "lora_alpha": 64}))  # = R01
    params = [p for p in model.parameters() if p.requires_grad]
    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=64, max_loras=1, gpu_memory_utilization=a.vllm_mem,
              max_model_len=2048 + a.max_completion, seed=a.seed, enable_prefix_caching=True,
              enable_sleep_mode=a.tok_budget > 0)
    from beat import fastlp
    adir = out / "adapter_at"; model.save_pretrained(str(adir))
    lreq = LoRARequest("diag", 1, str(adir))
    sp = SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion,
                        stop_token_ids=stop_ids, n=a.num_gen)
    eps_lo, eps_hi = GRPO["epsilon"], GRPO["epsilon_high"]

    def logps_of(ctxs_all, comps_all, grad):
        if a.tok_budget > 0:
            return fastlp.logps(model, tok.pad_token_id, ctxs_all, comps_all, grad,
                                a.tok_budget if grad else 2 * a.tok_budget)
        res = []
        for i in range(0, len(ctxs_all), a.micro_batch):
            ctxs, comps = ctxs_all[i:i + a.micro_batch], comps_all[i:i + a.micro_batch]
            seqs = [c + d for c, d in zip(ctxs, comps)]
            L, K = max(map(len, seqs)), max(map(len, comps)) + 1
            ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, L - len(s):] = torch.tensor(s); att[j, L - len(s):] = 1
            ids, att = ids.cuda(), att.cuda()
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            with torch.set_grad_enabled(grad):
                logits = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=K).logits
                for j, d in enumerate(comps):
                    n = len(d)
                    res.append(torch.log_softmax(logits[j, K - 1 - n:K - 1].float(), -1)
                               .gather(-1, ids[j, L - n:][:, None]).squeeze(-1))
            del logits
        return res

    def grad_of(smps, kind, tkey=None, mask=None, w_lo=0.5, w_hi=2.0, w_sign=1.0):
        """Flattened gradient (fp32, on GPU) of the token-mean loss; kind 'grpo' | 'sd' | 'tokw'."""
        for p in params:
            p.grad = None
        if not smps:
            return None
        if kind in ("sd", "tokw") and tkey is not None:
            model.eval()
            lt = logps_of([rows[s["i"]][tkey] for s in smps], [s["comp"] for s in smps], grad=False)
            for s, t in zip(smps, lt):
                s["_lpT"] = t.detach()
        model.train()
        ntok = sum(len(s["comp"]) for s in smps)
        if a.tok_budget > 0:
            groups = [[smps[i] for i in b] for b in
                      fastlp.token_batches([len(s["ctx"]) + len(s["comp"]) for s in smps], a.tok_budget)]
        else:
            groups = [smps[i:i + a.micro_batch] for i in range(0, len(smps), a.micro_batch)]
        for mb in groups:
            new = logps_of([s["ctx"] for s in mb], [s["comp"] for s in mb], grad=True)
            loss = 0.0
            for s, lp in zip(mb, new):
                if kind == "sd":
                    m = mask(s) if mask else torch.ones_like(lp)
                    loss = loss + (m * (lp.detach() - s["_lpT"]) * lp).sum()
                else:
                    adv = torch.full_like(lp, s["A"])
                    if kind == "tokw":   # RLSD: w = clip(exp(sign(A) * (logp_T - logp_S)), .5, 2), mean 1
                        w = ((s["_lpT"] - s["old"]) * (1.0 if s["A"] > 0 else -1.0) * w_sign).exp().clamp(w_lo, w_hi)
                        adv = adv * (w / w.mean())
                    ratio = (lp - s["old"]).exp()   # = 1 at the measured point; kept for parity with loop.py
                    loss = loss - torch.minimum(ratio * adv, ratio.clamp(1 - eps_lo, 1 + eps_hi) * adv).sum()
            (loss / ntok).backward()
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
        for p in params:
            p.grad = None
        return g

    def cos(u, v):
        return None if u is None or v is None else float(u @ v / (u.norm() * v.norm() + 1e-12))

    def pre_boxed(s):
        comp = s["comp"]; cut = next((k for k, t in enumerate(comp) if t in BOXED), len(comp))
        m = torch.zeros(len(comp), device="cuda"); m[:max(cut - 2, 0)] = 1
        return m

    import re as _re
    # y* token classes: prose (words outside math = the plan), math with digits (computed values), math symbols, answer box
    MATH = _re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|\$(.*?)\$", _re.S)
    KEYS = ("prose", "mnum", "msym", "box")

    def classes(qi):
        ref = rows[qi]["ref_txt"]
        enc = tok(ref, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]; n = len(rows[qi]["ref_ids"])
        assert len(offs) == n - 1, (len(offs), n)
        spans = [(m.start(), m.end()) for m in MATH.finditer(ref)]
        bx = ref.rfind("\\boxed")
        M = {k: torch.zeros(n, device="cuda") for k in KEYS}
        for k, (u, v) in enumerate(offs):
            piece = ref[u:v]
            if bx >= 0 and u >= bx - 1:
                M["box"][k] = 1
            elif any(u < e and v > st for st, e in spans):
                M["mnum" if any(ch.isdigit() for ch in piece) else "msym"][k] = 1
            else:
                M["prose"][k] = 1
        M["box"][n - 1] = 1          # eos belongs to the answer part
        M["all"] = torch.ones(n, device="cuda")
        return M

    def nll_grad(items):
        """items: (ctx, comp, mask). Gradient of the masked token-mean NLL; also returns mean NLL per item."""
        for p in params:
            p.grad = None
        model.train()
        ntok = sum(float(m.sum()) for _, _, m in items)
        if ntok == 0:
            return None, []
        nlls = []
        for k0 in range(0, len(items), 8):
            ch = items[k0:k0 + 8]
            lps = logps_of([c for c, _, _ in ch], [d for _, d, _ in ch], True)
            loss = 0.0
            for (_, _, m), lp in zip(ch, lps):
                loss = loss - (m * lp).sum()
                if m.sum() > 0:
                    nlls.append(float(-(m * lp.detach()).sum() / m.sum()))
            (loss / ntok).backward()
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
        for p in params:
            p.grad = None
        return g, nlls

    order = np.random.permutation(len(rows))
    results, token_rows = [], []
    for b in range(a.batches):
        t0 = time.time()
        idx = order[b * a.prompts:(b + 1) * a.prompts].tolist()
        if a.tok_budget > 0 and b > 0:
            torch.cuda.empty_cache(); llm.wake_up()
        outs = llm.generate([{"prompt_token_ids": rows[i]["p_ids"]} for i in idx], sp, lora_request=lreq, use_tqdm=False)
        smps = []
        for g, (i, o) in enumerate(zip(idx, outs)):
            for c in o.outputs:
                comp = list(c.token_ids)
                if c.finish_reason == "stop" and (not comp or comp[-1] not in stop_ids):
                    comp.append(eos_id)
                smps.append({"g": g, "i": i, "ctx": rows[i]["p_ids"], "comp": comp, "text": c.text,
                             "trunc": c.finish_reason == "length", "half": "A" if g < len(idx) // 2 else "B"})
        if a.tok_budget > 0:
            llm.sleep(level=1)
        rew = score_all([s["text"] for s in smps], [rows[s["i"]]["gt"] for s in smps], 8)
        G = a.num_gen
        gr = np.array(rew).reshape(len(idx), G)
        for s, r in zip(smps, rew):
            s["r"] = r; s["A"] = r - float(gr[s["g"]].mean())
        smps = [s for s in smps if not s["trunc"]]
        model.eval()
        olds = logps_of([s["ctx"] for s in smps], [s["comp"] for s in smps], grad=False)
        for s, o in zip(smps, olds):
            s["old"] = o.detach()
        kind_g = lambda s: "allfail" if gr[s["g"]].sum() == 0 else "allcorrect" if gr[s["g"]].sum() == G else "mixed"
        A = [s for s in smps if s["half"] == "A"]; B = [s for s in smps if s["half"] == "B"]
        cap = lambda L: random.sample(L, min(len(L), a.max_support))
        rlA, rlB = [s for s in A if s["A"] != 0], [s for s in B if s["A"] != 0]
        # fix every sample subset ONCE per batch, so theta vs theta' comparisons differ only by the parameters
        rlA_s, rlB_s = cap(rlA), cap(rlB)
        gB = grad_of(rlB_s, "grpo")
        gA = grad_of(rlA_s, "grpo")
        res = {"batch": b, "reward": float(gr.mean()), "allfail": float((gr.sum(1) == 0).mean()),
               "n": {"rl_A": len(rlA), "rl_B": len(rlB)}}
        res["anchor_grpoA_vs_grpoB"] = cos(gA, gB)
        if a.adapted_sweep:
            af0 = [s for s in A if kind_g(s) == "allfail"]
            mw0 = [s for s in A if kind_g(s) == "mixed" and s["r"] == 0]
            sup = {"allfail": cap(af0), "mixedwrong": cap(mw0)}

            def lr_stats(S, tag):
                d = torch.cat([(x["_lpT"] - x["old"]).float() for x in S])
                res[f"{tag}_lr_abs"] = float(d.abs().mean())
                res[f"{tag}_big"] = float((d.abs() > 1).float().mean())
                pos_tail = []
                for x in S:                       # share of the positive gap from the first "boxed" token on (answer leak)
                    dd = (x["_lpT"] - x["old"]).clamp(min=0).cpu()
                    cut = next((k for k, t in enumerate(x["comp"]) if t in BOXED), len(x["comp"]))
                    pos_tail.append(float(dd[max(cut - 2, 0):].sum()) / (float(dd.sum()) + 1e-9))
                res[f"{tag}_gap_from_boxed"] = float(np.mean(pos_tail))

            def adapted_teacher(S, eps):
                """_lpT of every sample under theta'_x = theta - eps * g_x/|g_x|, g_x = grad of NLL(y*_x | x) on the LoRA."""
                theta0 = [p.detach().clone() for p in params]
                byq = {}
                for x in S:
                    byq.setdefault(x["i"], []).append(x)
                for qi, xs in byq.items():
                    for p in params:
                        p.grad = None
                    model.train()
                    lp = logps_of([rows[qi]["p_ids"]], [rows[qi]["ref_ids"]], True)[0]
                    (-lp.mean()).backward()
                    g = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                    gn = torch.sqrt(sum((x_.float() ** 2).sum() for x_ in g)) + 1e-12
                    with torch.no_grad():
                        for p, gi in zip(params, g):
                            p.add_(-(eps / gn) * gi)
                    model.eval()
                    with torch.no_grad():
                        lt = logps_of([x["ctx"] for x in xs], [x["comp"] for x in xs], False)
                    for x, t in zip(xs, lt):
                        x["_lpT"] = t.detach()
                    with torch.no_grad():
                        for p, t in zip(params, theta0):
                            p.copy_(t)
                for p in params:
                    p.grad = None

            # reference: the plain SFT gradient on y* of the A questions (the "SFT on Luna" direction), vs reward on B
            qA = sorted({x["i"] for x in A})[:a.max_support]
            for p in params:
                p.grad = None
            model.train()
            ntok = sum(len(rows[qi]["ref_ids"]) for qi in qA)
            for k0 in range(0, len(qA), 8):
                lps = logps_of([rows[qi]["p_ids"] for qi in qA[k0:k0 + 8]], [rows[qi]["ref_ids"] for qi in qA[k0:k0 + 8]], True)
                (-sum(l.sum() for l in lps) / ntok).backward()
            gSFT = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
            for p in params:
                p.grad = None
            res["sft_ystar_cos"] = cos(gSFT, gB)
            for sname, S in sup.items():
                if not S:
                    continue
                for fname, key in (("ctx_concise", "tp_ids"), ("ctx_full", "hp_ids")):
                    res[f"{fname}_{sname}_cos"] = cos(grad_of(S, "sd", key), gB)
                    lr_stats(S, f"{fname}_{sname}")
                for eps in map(float, a.adapt_eps.split(",")):
                    adapted_teacher(S, eps)
                    tag = f"wt{eps:g}_{sname}"
                    lr_stats(S, tag)                       # stats before grad_of (which keeps _lpT as is)
                    res[f"{tag}_cos"] = cos(grad_of(S, "sd", None), gB)
            res["time"] = round(time.time() - t0, 1)
            results.append(res); print("BATCH " + json.dumps(res), flush=True)
            continue
        if a.dr_sweep:
            from torch.utils.checkpoint import checkpoint as _ckpt
            base_ = model.get_base_model(); body_, head_ = base_.model, base_.lm_head

            def _cv_chunk(hs, ht, tg):
                lS = torch.log_softmax(head_(hs).float(), -1)
                with torch.no_grad():
                    phi = torch.log_softmax(head_(ht).float(), -1) - lS.detach()
                samp = (phi.gather(-1, tg[:, None])[:, 0] * lS.gather(-1, tg[:, None])[:, 0]).sum()
                return samp - (lS.exp() * phi).sum(), phi.gather(-1, tg[:, None])[:, 0].sum().detach()

            def cv_grad(S, tkey):
                """grad of sum_t [phi_t(a_t) log pi(a_t) - sum_a pi(a) phi_t(a)] / ntok, phi = log p_T - log p_S (detached).
                Zero in expectation over the student's own sampling; returns (gradient, mean phi(a_t))."""
                for p in params:
                    p.grad = None
                model.train()
                ntok = sum(len(x["comp"]) for x in S); phis = []
                for x in S:
                    n = len(x["comp"]); comp = torch.tensor(x["comp"], device="cuda")
                    tctx = rows[x["i"]][tkey]
                    with torch.no_grad():
                        ht = body_(input_ids=torch.tensor([tctx + x["comp"]], device="cuda"), use_cache=False).last_hidden_state[0]
                        ht = ht[len(tctx) - 1:len(tctx) - 1 + n]
                    hs = body_(input_ids=torch.tensor([x["ctx"] + x["comp"]], device="cuda"), use_cache=False).last_hidden_state[0]
                    hs = hs[len(x["ctx"]) - 1:len(x["ctx"]) - 1 + n]
                    tot = 0.0
                    for s0 in range(0, n, 512):
                        term, ph = _ckpt(_cv_chunk, hs[s0:s0 + 512], ht[s0:s0 + 512], comp[s0:s0 + 512], use_reentrant=False)
                        tot = tot + term; phis.append(float(ph))
                    (tot / ntok).backward()
                    del ht, hs
                g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])
                for p in params:
                    p.grad = None
                return g, sum(phis) / ntok

            SA = random.sample(A, min(len(A), a.dr_samples)); SB = random.sample(B, min(len(B), a.dr_samples))
            gGA, gGB = grad_of(SA, "grpo"), grad_of(SB, "grpo")
            res["dr_grpo_AB"] = cos(gGA, gGB)          # anchor on these sets (A = 0 samples included)
            for tname, tkey in (("full", "hp_ids"), ("concise", "tp_ids")):
                cA, phA = cv_grad(SA, tkey); cB, phB = cv_grad(SB, tkey)
                res[f"dr_{tname}_phi_mean"] = (phA + phB) / 2
                res[f"dr_{tname}_cvnorm_ratio"] = float((cA.norm() / (gGA.norm() + 1e-12) + cB.norm() / (gGB.norm() + 1e-12)) / 2)
                # anchors: the control variate has zero mean -> no alignment across halves or with the reward gradient
                res[f"dr_{tname}_cv_AB"] = cos(cA, cB)
                res[f"dr_{tname}_cv_vs_grpoB"] = cos(cA, gGB)
                res[f"dr_{tname}_cv_vs_grpoA_same"] = cos(cA, gGA)   # same samples: negative => variance can be cut
                for beta in (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
                    dA, dB = gGA + beta * cA, gGB + beta * cB
                    res[f"dr_{tname}_b{beta:g}_AB"] = cos(dA, dB)
                    res[f"dr_{tname}_b{beta:g}_vs_grpoB"] = cos(dA, gGB)
                bA = -float(gGA @ cA) / float(cA @ cA + 1e-20); bB = -float(gGB @ cB) / float(cB @ cB + 1e-20)
                res[f"dr_{tname}_bstar"] = (bA + bB) / 2
                dA, dB = gGA + bA * cA, gGB + bB * cB
                res[f"dr_{tname}_bstar_AB"] = cos(dA, dB)
                res[f"dr_{tname}_bstar_normratio"] = float((dA.norm() / gGA.norm() + dB.norm() / gGB.norm()) / 2)
                # cross-fitted beta: the coefficient fitted on one half is applied to the other half
                res[f"dr_{tname}_bcross_AB"] = cos(gGA + bB * cA, gGB + bA * cB)
                res[f"dr_{tname}_bcross_vs_grpoB"] = cos(gGA + bB * cA, gGB)
            res["n"].update({"dr_A": len(SA), "dr_B": len(SB)})
            res["time"] = round(time.time() - t0, 1)
            results.append(res); print("BATCH " + json.dumps(res), flush=True)
            continue
        if a.causal_sweep:
            rng = random.Random(a.seed * 1000 + b)

            def perturb(ref, kind, question=""):
                """Minimal same-style counterfactual of y*: 1-2 DERIVED values (numbers absent from the question) or 1-2 +/- signs
                inside formulas are changed; 'shuffle' permutes lines (anchor: destroys the argument, keeps the tokens)."""
                bx = ref.rfind("\\boxed"); bx = len(ref) if bx < 0 else bx
                if kind == "shuffle":
                    lines = ref[:bx].split("\n"); rng.shuffle(lines); return "\n".join(lines) + ref[bx:]
                spans = [(m.start(), min(m.end(), bx)) for m in MATH.finditer(ref) if m.start() < bx]
                given = set(_re.findall(r"\d+", question))
                if kind == "numbers":
                    cand = [(st_ + u_, st_ + v_) for (st_, en_) in spans for (u_, v_) in
                            [(m.start(), m.end()) for m in _re.finditer(r"\d+", ref[st_:en_]) if m.group() not in given]]
                else:
                    cand = [(st_ + m.start(), st_ + m.end()) for st_, en_ in spans for m in _re.finditer(r"[+-]", ref[st_:en_])]
                if not cand:
                    return ref
                pick = sorted(rng.sample(cand, min(len(cand), rng.choice([1, 2]))))
                out, last = [], 0
                for u_, v_ in pick:
                    tok_ = ref[u_:v_]
                    if kind == "numbers":
                        n_ = int(tok_); new_ = str(n_ + rng.choice([1, 2, 3]) if n_ < 3 else n_ + rng.choice([-2, -1, 1, 2, 3]))
                    else:
                        new_ = "-" if tok_ == "+" else "+"
                    out += [ref[last:u_], new_]; last = v_
                out.append(ref[last:]); return "".join(out)

            qA = sorted({x["i"] for x in A})[:a.max_support]
            ones = lambda ids: torch.ones(len(ids), device="cuda")
            g_y, nl_y = nll_grad([(rows[q]["p_ids"], rows[q]["ref_ids"], ones(rows[q]["ref_ids"])) for q in qA])
            res["cf_ystar_cos"] = cos(g_y, gB)
            # R03-1: grad of [NLL(y*) - NLL(y~*)] (descent raises log p(y*) - log p(y~*)); style shared by both cancels
            for kind in ("numbers", "operators", "shuffle"):
                cf = {}
                for q in qA:
                    t = perturb(rows[q]["ref_txt"], kind, rows[q]["q_txt"])
                    if t != rows[q]["ref_txt"]:
                        cf[q] = tok(t, add_special_tokens=False)["input_ids"] + [eos_id]
                Q = [q for q in qA if q in cf]
                if len(Q) < 8:
                    continue
                g1, n1 = nll_grad([(rows[q]["p_ids"], rows[q]["ref_ids"], ones(rows[q]["ref_ids"])) for q in Q])
                g2, n2 = nll_grad([(rows[q]["p_ids"], cf[q], ones(cf[q])) for q in Q])
                res[f"cf_{kind}_cos"] = cos(g1 - g2, gB)
                res[f"cf_{kind}_margin"] = float(np.mean(np.array(n2) - np.array(n1)))   # > 0: model already prefers y*
                res[f"cf_{kind}_vs_ystar"] = cos(g1 - g2, g_y)
                res["n"][f"cf_{kind}"] = len(Q)
            # R03-2: invariance across voices. env1 = y*, env2 = own correct solution (same questions), env3 = GRPO on A
            own = {}
            for x in A:
                if x["r"] == 1 and x["i"] not in own:
                    own[x["i"]] = x
            Qb = [q for q in qA if q in own]
            gy, _ = nll_grad([(rows[q]["p_ids"], rows[q]["ref_ids"], ones(rows[q]["ref_ids"])) for q in Qb])
            go, _ = nll_grad([(own[q]["ctx"], own[q]["comp"], ones(own[q]["comp"])) for q in Qb])
            u = lambda g: g / (g.norm() + 1e-12)
            gy_, go_, gA_ = u(gy), u(go), u(gA)
            res["inv_own_cos"] = cos(go, gB); res["inv_ystar_cos"] = cos(gy, gB); res["inv_ystar_vs_own"] = cos(gy, go)
            res["inv_mean2_cos"] = cos(gy_ + go_, gB)
            m2 = (torch.sign(gy) == torch.sign(go)) & (gy != 0)
            res["inv_and2_cos"] = cos(m2 * (gy_ + go_), gB)
            res["inv_soft2_cos"] = cos(m2 * torch.sign(gy) * torch.sqrt(gy_.abs() * go_.abs()), gB)
            res["inv_and2_density"] = float(m2.float().sum() / (gy != 0).float().sum())
            gen = torch.Generator(device="cuda").manual_seed(a.seed + 17 * b)
            rmask2 = (torch.rand(gy.shape, device="cuda", generator=gen) < res["inv_and2_density"]) & (gy != 0)
            res["inv_rand2_cos"] = cos(rmask2 * (gy_ + go_), gB)
            m3 = m2 & (torch.sign(gA) == torch.sign(gy))
            res["inv_and3_cos"] = cos(m3 * (gy_ + go_ + gA_), gB)
            res["inv_and3_density"] = float(m3.float().sum() / (gy != 0).float().sum())
            rmask3 = (torch.rand(gy.shape, device="cuda", generator=gen) < res["inv_and3_density"]) & (gy != 0)
            res["inv_rand3_cos"] = cos(rmask3 * (gy_ + go_ + gA_), gB)
            res["inv_grpo_masked_cos"] = cos(m3 * gA_, gB)          # GRPO kept only where both voices agree with it
            res["inv_grpo_rand_cos"] = cos(rmask3 * gA_, gB)
            res["n"]["inv_q"] = len(Qb)
            # placebo-differenced teacher: log pT(.|x, y*) - log pT(.|x, y* of another question), on incorrect samples of mixed groups
            mw = cap([s_ for s_ in A if kind_g(s_) == "mixed" and s_["r"] == 0])
            if mw:
                others = [r_ for r_ in range(len(rows))]
                plac = {}
                for x in mw:
                    if x["i"] not in plac:
                        j = rng.choice(others)
                        while j == x["i"]:
                            j = rng.choice(others)
                        plac[x["i"]] = tok(tok.apply_chat_template(
                            [{"role": "system", "content": rows[x["i"]]["sys_txt"]},
                             {"role": "user", "content": HINT_TEMPLATE.format(question=rows[x["i"]]["q_txt"],
                                                                              reference=rows[j]["ref_txt"])}],
                            tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
                model.eval()
                with torch.no_grad():
                    lt_real = logps_of([rows[x["i"]]["hp_ids"] for x in mw], [x["comp"] for x in mw], False)
                    lt_plac = logps_of([plac[x["i"]] for x in mw], [x["comp"] for x in mw], False)
                for x, t in zip(mw, lt_real):
                    x["_lpT"] = t.detach()
                res["pl_real_cos"] = cos(grad_of(mw, "sd", None), gB)
                for x, t in zip(mw, lt_plac):
                    x["_lpT"] = t.detach()
                res["pl_placebo_only_cos"] = cos(grad_of(mw, "sd", None), gB)
                for x, tr, tp in zip(mw, lt_real, lt_plac):
                    x["_lpT"] = x["old"] + (tr - tp).detach()
                res["pl_diff_cos"] = cos(grad_of(mw, "sd", None), gB)
                d_real = torch.cat([(tr - x["old"]).float() for x, tr in zip(mw, lt_real)])
                d_diff = torch.cat([(tr - tp).float() for tr, tp in zip(lt_real, lt_plac)])
                res["pl_real_big"] = float((d_real.abs() > 1).float().mean()); res["pl_diff_big"] = float((d_diff.abs() > 1).float().mean())
                res["n"]["pl"] = len(mw)
            res["time"] = round(time.time() - t0, 1)
            results.append(res); print("BATCH " + json.dumps(res), flush=True)
            continue
        if a.ystar_split:
            qA = sorted({x["i"] for x in A})[:a.max_support]
            MS = {qi: classes(qi) for qi in qA}
            for k in KEYS:
                res[f"ys_frac_{k}"] = float(np.mean([float(MS[q][k].sum() / len(MS[q][k])) for q in qA]))
            gc = {}
            for k in ("all",) + KEYS:
                g, nl = nll_grad([(rows[q]["p_ids"], rows[q]["ref_ids"], MS[q][k]) for q in qA])
                gc[k] = g
                res[f"ys_{k}_cos"] = cos(g, gB)
                res[f"ys_{k}_nll"] = float(np.mean(nl)) if nl else None
            res["ys_prose_vs_mnum_cos"] = cos(gc["prose"], gc["mnum"])
            # y* length split (short = most compressed half)
            Ls = sorted(qA, key=lambda q: len(rows[q]["ref_ids"])); h = len(Ls) // 2
            for tag, Q in (("short", Ls[:h]), ("long", Ls[h:])):
                g, _ = nll_grad([(rows[q]["p_ids"], rows[q]["ref_ids"], MS[q]["all"]) for q in Q])
                res[f"ys_len_{tag}_cos"] = cos(g, gB)
            res["ys_len_median"] = float(len(rows[Ls[h]]["ref_ids"]))
            # anchor: SFT on the student's OWN correct solutions (A half), one per question, same number of questions
            own, seen = [], set()
            for x in A:
                if x["r"] == 1 and x["i"] not in seen:
                    seen.add(x["i"]); own.append(x)
            own = own[:len(qA)]
            g_own, nl_own = nll_grad([(x["ctx"], x["comp"], torch.ones(len(x["comp"]), device="cuda")) for x in own])
            res["ys_own_cos"] = cos(g_own, gB); res["ys_own_nll"] = float(np.mean(nl_own)) if nl_own else None
            res["ys_own_vs_ystar_cos"] = cos(g_own, gc["all"])
            res["ys_own_len_median"] = float(np.median([len(x["comp"]) for x in own])) if own else None
            res["n"].update({"ystar_q": len(qA), "own_q": len(own)})
            res["time"] = round(time.time() - t0, 1)
            results.append(res); print("BATCH " + json.dumps(res), flush=True)
            continue
        if a.skeleton_sweep:
            qA = sorted({x["i"] for x in A})[:a.max_support]
            taus = [float(v) for v in a.skel_taus.split(",") if v]
            eos_tok = tok.convert_tokens_to_ids(tok.eos_token)

            def steps_of(ref):
                """Split y* into steps at sentence ends / line breaks outside math; the step holding \\boxed and all after = answer."""
                spans = [(m.start(), m.end()) for m in MATH.finditer(ref)]
                inside = lambda c: any(st < c < e for st, e in spans)
                cuts = [m.end() for m in _re.finditer(r"(?<=[.;:])\s+|\n+", ref) if not inside(m.start())]
                bounds = [0] + [c for c in cuts if 0 < c < len(ref)] + [len(ref)]
                st = [ref[u:v] for u, v in zip(bounds, bounds[1:]) if ref[u:v]]
                bx = next((k for k, t in enumerate(st) if "\\boxed" in t), len(st) - 1)
                return st[:bx] + ["".join(st[bx:])]           # last element = answer node

            SK, stats = {}, {"n_steps": [], "edges": {t: [] for t in taus}, "skel_frac": {t: [] for t in taus},
                             "prose_in_skel": {t: [] for t in taus}, "prose_in_rest": {t: [] for t in taus},
                             "direct_ans_drop": []}
            model.eval()
            for qi in qA:
                st = steps_of(rows[qi]["ref_txt"])
                tks, cont = [], []
                for k, t in enumerate(st):
                    e = tok(t, add_special_tokens=False, return_offsets_mapping=True)
                    sp_t = [(m.start(), m.end()) for m in MATH.finditer(t)]
                    tks.append(e["input_ids"])
                    # content tokens: digits inside math (computed values); the answer node counts entirely
                    cont.append([k == len(st) - 1 or (any(ch.isdigit() for ch in t[u:v]) and any(u < e2 and v > s2 for s2, e2 in sp_t))
                                 for u, v in e["offset_mapping"]])
                n = len(st) - 1                               # deletable steps
                if n < 2:
                    continue
                full = [t for tt in tks for t in tt] + [eos_tok]
                comps = [full] + [[t for k, tt in enumerate(tks) if k != d for t in tt] + [eos_tok] for d in range(n)]
                with torch.no_grad():
                    lps = [lp.float().cpu() for lp in logps_of([rows[qi]["p_ids"]] * len(comps), comps, grad=False)]
                off = np.cumsum([0] + [len(tt) for tt in tks])  # token offset of each step in full
                # drop[d][j] for j > d: mean drop of log-prob over the CONTENT tokens of step j (j = n: answer node incl. eos).
                # Measuring all tokens would mostly pick up fluency (any deleted sentence makes the next one less predictable).
                drop = np.zeros((n, n + 1))
                for d in range(n):
                    shift = len(tks[d])
                    for j in range(d + 1, n + 1):
                        cm = torch.tensor(cont[j] + ([True] if j == n else []), dtype=torch.bool)
                        if not bool(cm.any()):
                            continue
                        L = len(cm)
                        drop[d, j] = float((lps[0][off[j]:off[j] + L] - lps[d + 1][off[j] - shift:off[j] - shift + L])[cm].mean())
                stats["n_steps"].append(n); stats["direct_ans_drop"].append(float(drop[:, n].mean()))
                M = classes(qi) if full == rows[qi]["ref_ids"] else None
                SK[qi] = {"full": full, "tks": tks, "n": n, "masks": {}}
                for tau in taus:
                    anc, frontier = {n}, [n]
                    while frontier:
                        j = frontier.pop()
                        for d in range(min(j, n)):
                            if d not in anc and drop[d, j] > tau:
                                anc.add(d); frontier.append(d)
                    keep = torch.zeros(len(full), device="cuda"); ans = torch.zeros(len(full), device="cuda")
                    for k in range(n + 1):
                        if k in anc:
                            keep[off[k]:off[k] + len(tks[k])] = 1
                    ans[off[n]:] = 1; keep[-1] = 1
                    SK[qi]["masks"][tau] = (keep, ans, sorted(anc - {n}))
                    stats["edges"][tau].append(float((drop[:, :] > tau).sum()) / max(1, n * (n + 1) / 2))
                    nonans = 1 - ans
                    stats["skel_frac"][tau].append(float((keep * nonans).sum() / nonans.sum().clamp(min=1)))
                    if M is not None:
                        pr = M["prose"] * nonans
                        stats["prose_in_skel"][tau].append(float((pr * keep).sum() / (keep * nonans).sum().clamp(min=1)))
                        stats["prose_in_rest"][tau].append(float((pr * (1 - keep)).sum() / (nonans * (1 - keep)).sum().clamp(min=1)))
            Qs = sorted(SK)
            res["sk_q"] = len(Qs)
            if not Qs:
                results.append(res); print("BATCH " + json.dumps(res), flush=True)
                continue
            res["sk_n_steps"] = float(np.mean(stats["n_steps"])); res["sk_direct_ans_drop"] = float(np.mean(stats["direct_ans_drop"]))
            ones = lambda q: torch.ones(len(SK[q]["full"]), device="cuda")
            g_all, _ = nll_grad([(rows[q]["p_ids"], SK[q]["full"], ones(q)) for q in Qs]); res["sk_all_cos"] = cos(g_all, gB)
            g_ans, _ = nll_grad([(rows[q]["p_ids"], SK[q]["full"], SK[q]["masks"][taus[0]][1]) for q in Qs])
            res["sk_answer_only_cos"] = cos(g_ans, gB)
            rng_m = np.random.default_rng(a.seed + b)
            for tau in taus:
                tg = f"{tau:g}"
                res[f"sk_frac_{tg}"] = float(np.mean(stats["skel_frac"][tau])); res[f"sk_edges_{tg}"] = float(np.mean(stats["edges"][tau]))
                res[f"sk_prose_skel_{tg}"] = float(np.mean(stats["prose_in_skel"][tau])) if stats["prose_in_skel"][tau] else None
                res[f"sk_prose_rest_{tg}"] = float(np.mean(stats["prose_in_rest"][tau])) if stats["prose_in_rest"][tau] else None
                g, _ = nll_grad([(rows[q]["p_ids"], SK[q]["full"], SK[q]["masks"][tau][0]) for q in Qs])
                res[f"sk_skel_cos_{tg}"] = cos(g, gB)
                g, _ = nll_grad([(rows[q]["p_ids"], SK[q]["full"], 1 - SK[q]["masks"][tau][0]) for q in Qs])
                res[f"sk_rest_cos_{tg}"] = cos(g, gB)
                g, _ = nll_grad([(rows[q]["p_ids"], SK[q]["full"], SK[q]["masks"][tau][0] * (1 - SK[q]["masks"][tau][1])) for q in Qs])
                res[f"sk_skel_noans_cos_{tg}"] = cos(g, gB)
                # control: random steps with the same number of kept steps, + the answer node
                items = []
                for q in Qs:
                    keep, ansm, anc = SK[q]["masks"][tau]; n = SK[q]["n"]; tks = SK[q]["tks"]
                    pick = set(rng_m.choice(n, size=len(anc), replace=False).tolist()) if anc else set()
                    m = ansm.clone(); o = 0
                    for k in range(n):
                        if k in pick:
                            m[o:o + len(tks[k])] = 1
                        o += len(tks[k])
                    items.append((rows[q]["p_ids"], SK[q]["full"], m))
                g, _ = nll_grad(items); res[f"sk_randsteps_cos_{tg}"] = cos(g, gB)
                # intervened data: the skeleton TEXT alone (non-ancestor steps deleted), full SFT on it
                items = []
                for q in Qs:
                    anc = set(SK[q]["masks"][tau][2]); tks = SK[q]["tks"]; n = SK[q]["n"]
                    comp = [t for k in range(n + 1) if (k in anc or k == n) for t in tks[k]] + [eos_tok]
                    items.append((rows[q]["p_ids"], comp, torch.ones(len(comp), device="cuda")))
                g, _ = nll_grad(items); res[f"sk_skeltext_cos_{tg}"] = cos(g, gB)
            ex = Qs[0]
            res["example_skel"] = {"steps": [t[:120] for t in steps_of(rows[ex]["ref_txt"])],
                                 "ancestors": {f"{t:g}": SK[ex]["masks"][t][2] for t in taus}}
            res["time"] = round(time.time() - t0, 1)
            results.append(res); print("BATCH " + json.dumps(res), flush=True)
            continue
        if a.format_sweep:
            af0 = [s for s in A if kind_g(s) == "allfail"]
            mw0 = [s for s in A if kind_g(s) == "mixed" and s["r"] == 0]
            sup = {"allfail": cap(af0), "mixedwrong": cap(mw0)}
            for fname, key in (("concise", "tp_ids"), ("full", "hp_ids"), ("answer", "ap_ids"), ("multiturn", "mt_ids"),
                               ("system", "sys_ids"), ("selfprefix", "pre_ids")):
                for sname, S in sup.items():
                    g = grad_of(S, "sd", key)
                    res[f"fmt_{fname}_{sname}_cos"] = cos(g, gB)
                    if S:
                        d = torch.cat([(x["_lpT"] - x["old"]).float() for x in S])
                        res[f"fmt_{fname}_{sname}_lr_mean"] = float(d.mean())          # E[log pT - log pS] on student tokens
                        res[f"fmt_{fname}_{sname}_lr_abs"] = float(d.abs().mean())
                        res[f"fmt_{fname}_{sname}_big"] = float((d.abs() > 1).float().mean())
            res["time"] = round(time.time() - t0, 1)
            results.append(res); print("BATCH " + json.dumps(res), flush=True)
            continue
        if a.tokw_sweep:
            for lo, hi in ((1.0, 1.0), (0.9, 1.1), (0.8, 1.25), (0.7, 1.4), (0.5, 2.0)):
                res[f"tokw_{lo}_{hi}"] = cos(grad_of(rlA_s, "tokw", "tp_ids", w_lo=lo, w_hi=hi), gB)
            res["tokw_inverted_0.5_2.0"] = cos(grad_of(rlA_s, "tokw", "tp_ids", w_sign=-1.0), gB)
            res["time"] = round(time.time() - t0, 1)
            results.append(res); print("BATCH " + json.dumps(res), flush=True)
            continue
        af = [s for s in A if kind_g(s) == "allfail"]
        mw = [s for s in A if kind_g(s) == "mixed" and s["r"] == 0]
        mc = [s for s in A if kind_g(s) == "mixed" and s["r"] == 1]
        res["n"].update({"allfail": len(af), "mixed_wrong": len(mw), "mixed_correct": len(mc)})
        af_s, mw_s, mc_s = cap(af), cap(mw), cap(mc)
        gS = grad_of(af_s, "sd", "tp_ids")
        res["sd_allfail"] = cos(gS, gB)
        res["sd_mixed_wrong"] = cos(grad_of(mw_s, "sd", "tp_ids"), gB)
        res["sd_correct"] = cos(grad_of(mc_s, "sd", "tp_ids"), gB)
        res["sd_allfail_ans"] = cos(grad_of(af_s, "sd", "ap_ids"), gB)
        res["sd_allfail_full"] = cos(grad_of(af_s, "sd", "hp_ids"), gB)
        res["sd_allfail_pre"] = cos(grad_of(af_s, "sd", "tp_ids", mask=pre_boxed), gB)
        res["grpo_tokw"] = cos(grad_of(rlA_s, "tokw", "tp_ids"), gB)

        # R01's own measurement: g_Q taken at theta' = theta - 0.1 * d/|d| (after an inner step along d), not at theta.
        # Controls with the same step norm: d = a random direction on the same coordinates, and d = the GRPO gradient
        # on A. If the controls drop as much as the teacher step, the drop is curvature, not something the teacher does.
        def after_step(d):
            if d is None:
                return None, None
            theta0 = [p.detach().clone() for p in params]
            st = -0.1 * d / (d.norm() + 1e-12)
            off = 0
            with torch.no_grad():
                for p in params:
                    n = p.numel(); p.add_(st[off:off + n].view_as(p).to(p.dtype)); off += n
            g1 = grad_of(rlB_s, "grpo")
            with torch.no_grad():
                for p, t in zip(params, theta0):
                    p.copy_(t)
            return cos(g1, d), cos(g1, gB)
        gen = torch.Generator(device="cuda").manual_seed(a.seed + b)
        rnd = None if gS is None else torch.randn(gS.shape, device=gS.device, generator=gen) * (gS != 0)
        for name, d in (("teacher", gS), ("random", rnd), ("grpoA", gA)):
            c_d, c_b = after_step(d)
            res[f"step_{name}_cos_at_theta"] = cos(d, gB)
            res[f"step_{name}_cos_at_thetaprime"] = c_d
            res[f"step_{name}_gQshift"] = c_b
        # where does the teacher-student gap sit? (all-fail samples, teacher = TEACHER_TEMPLATE)
        sub = af[:64]
        if sub:
            model.eval()
            lt = logps_of([rows[s["i"]]["tp_ids"] for s in sub], [s["comp"] for s in sub], grad=False)
            ls = logps_of([s["ctx"] for s in sub], [s["comp"] for s in sub], grad=False)
            tail, boxed_share, tops = [], [], []
            for s, t, l in zip(sub, lt, ls):
                d = (t - l).cpu()                     # > 0: teacher wants this token more than the student does
                pos = d.clamp(min=0)
                n = len(d); tot = float(pos.sum()) + 1e-9
                tail.append(float(pos[int(0.9 * n):].sum()) / tot)
                cut = next((k for k, x in enumerate(s["comp"]) if x in BOXED), n)
                boxed_share.append(float(pos[max(cut - 2, 0):].sum()) / tot)
                for k in torch.topk(d, min(3, n)).indices.tolist():
                    tops.append(tok.decode(s["comp"][max(0, k - 6):k]) + " ⟦" + tok.decode([s["comp"][k]]) + "⟧")
            res["gap_share_last10pct"] = float(np.mean(tail))
            res["gap_share_from_boxed"] = float(np.mean(boxed_share))
            token_rows += tops
        res["time"] = round(time.time() - t0, 1)
        results.append(res)
        print("BATCH " + json.dumps(res), flush=True)

    keys = [k for k in results[0] if k.startswith(("anchor", "sd_", "grpo_", "gap_", "step_", "tokw_", "fmt_", "ctx_", "wt", "sft_", "ys_", "cf_", "inv_", "pl_", "dr_", "sk_"))]
    summ = {k: [r[k] for r in results if isinstance(r.get(k), (int, float))] for k in keys}
    summ = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in summ.items() if v}
    (out / "results.json").write_text(json.dumps({"batches": results, "summary": summ}, indent=2))
    (out / "teacher_top_tokens.txt").write_text("\n".join(token_rows))
    print("SUMMARY")
    for k, v in summ.items():
        print(f"  {k:24s} mean {v['mean']:+.3f}   [{v['min']:+.3f}, {v['max']:+.3f}]")
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
