"""Why did R04-1 (y*-teacher log-ratio as a control variate for GRPO) not reduce variance? Forward passes only.

phi_t(a) = log p_T(a | x, y*, prefix) - log p_S(a | x, prefix). A control variate helps only if phi_t(a_t) tracks the sample's
advantage. If the teacher were the success-conditioned policy pi(. | R = 1), Bayes gives phi_t(a_t) = log Q(s_t, a_t) - log V(s_t),
so the sum of phi over a correct solution should exceed that of a wrong one. Measured on base-model samples:
  1. within-group AUC (mixed groups) of mean / sum phi, before the final \\boxed and on the \\boxed region only;
  2. the same for a placebo teacher (y* of another question) -> is the information from y* or from the prompt?
  3. where phi sits: share of |phi| in the answer region, share of tokens with |phi| > 1, pooled corr(phi_t(a_t), A);
  4. size: per-token variance of the centred control variate phi_t(a_t) - E_pi[phi_t] vs E[A^2].

    PYTHONPATH=code python -m beat.diag_phi --prompts 256 --out runs/diag_phi
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=256)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--vllm-mem", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat.loop import HINT_TEMPLATE, TEACHER_TEMPLATE
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    eos_id = tok.convert_tokens_to_ids(tok.eos_token)
    BOXED = {i for i in range(len(tok)) if "boxed" in tok.decode([i])}

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        rows.append({"sys": msgs[0], "q": r.extra_info["question"], "ref": r.target[0]["content"], "gt": r.reward_model["ground_truth"],
                     "p": tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]})
    rows = [x for x in rows if len(x["p"]) <= 1024]
    idx = np.random.permutation(len(rows))[:a.prompts].tolist()
    enc = lambda x, tmpl, ref: tok(tok.apply_chat_template([x["sys"], {"role": "user", "content": tmpl.format(question=x["q"], reference=ref)}],
                                                           tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    for k, i in enumerate(idx):
        x = rows[i]; j = idx[(k + 1) % len(idx)]
        x["full"] = enc(x, HINT_TEMPLATE, x["ref"])
        x["concise"] = enc(x, TEACHER_TEMPLATE, x["ref"])
        x["placebo"] = enc(x, HINT_TEMPLATE, rows[j]["ref"])        # same prompt, y* of another question

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=a.vllm_mem, max_model_len=2048 + a.max_completion, seed=a.seed,
              enable_prefix_caching=True, enable_sleep_mode=True)
    sp = SamplingParams(temperature=GRPO["temperature"], top_p=1.0, max_tokens=a.max_completion, stop_token_ids=stop_ids, n=a.num_gen)
    outs = llm.generate([{"prompt_token_ids": rows[i]["p"]} for i in idx], sp, use_tqdm=False)
    llm.sleep(level=2)
    S = []
    for g, (i, o) in enumerate(zip(idx, outs)):
        for c in o.outputs:
            comp = list(c.token_ids)
            if c.finish_reason == "stop" and (not comp or comp[-1] not in stop_ids):
                comp.append(eos_id)
            S.append({"g": g, "i": i, "comp": comp, "text": c.text, "trunc": c.finish_reason == "length"})
    rew = score_all([s["text"] for s in S], [rows[s["i"]]["gt"] for s in S], 8)
    G = np.array(rew, dtype=float).reshape(len(idx), a.num_gen)
    for s, r in zip(S, rew):
        s["r"] = float(r); s["A"] = float(r) - float(G[s["g"]].mean())
    print(f"GEN {time.time() - t0:.0f}s reward {G.mean():.3f} allfail {(G.sum(1) == 0).mean():.3f}", flush=True)
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    body, head = model.model, model.lm_head

    @torch.no_grad()
    def hid(ctx, comp):
        h = body(input_ids=torch.tensor([ctx + comp], device="cuda"), use_cache=False).last_hidden_state[0]
        return h[len(ctx) - 1:len(ctx) - 1 + len(comp)]

    @torch.no_grad()
    def stats(hs, ht, comp):
        """per token: phi(a_t), E_pi[phi], KL(pi || p_T)."""
        tg = torch.tensor(comp, device="cuda")
        ph, eph = [], []
        for s0 in range(0, len(comp), 512):
            lS = torch.log_softmax(head(hs[s0:s0 + 512]).float(), -1)
            lT = torch.log_softmax(head(ht[s0:s0 + 512]).float(), -1)
            phi = lT - lS
            ph.append(phi.gather(-1, tg[s0:s0 + 512, None])[:, 0]); eph.append((lS.exp() * phi).sum(-1))
        return torch.cat(ph).cpu().numpy(), torch.cat(eph).cpu().numpy()

    TEACH = ("full", "concise", "placebo")
    for n_, s in enumerate(S):
        x = rows[s["i"]]
        hs = hid(x["p"], s["comp"])
        cut = next((k for k, t in enumerate(s["comp"]) if t in BOXED), len(s["comp"]))
        s["cut"] = max(cut - 2, 0)
        for tname in TEACH:
            ph, eph = stats(hs, hid(x[tname], s["comp"]), s["comp"])
            s[tname] = {"phi": ph, "eph": eph}
        if n_ % 512 == 0:
            print(f"FWD {n_}/{len(S)} {time.time() - t0:.0f}s", flush=True)

    def auc(score):
        num = den = 0.0
        for g in range(len(idx)):
            if not 0 < G[g].sum() < a.num_gen:
                continue
            ss = [s for s in S if s["g"] == g]
            c = [score(s) for s in ss if s["r"] == 1]; w = [score(s) for s in ss if s["r"] == 0]
            for u in c:
                for v in w:
                    num += 1.0 if u > v else 0.5 if u == v else 0.0; den += 1
        return num / den if den else None

    mean = lambda v: float(np.mean(v)) if len(v) else 0.0
    res = {"reward": float(G.mean()), "allfail": float((G.sum(1) == 0).mean()), "samples": len(S),
           "mixed_groups": int(sum(1 for g in range(len(idx)) if 0 < G[g].sum() < a.num_gen))}
    res["auc_neg_length"] = auc(lambda s: -len(s["comp"]))
    mixed = [s for s in S if 0 < G[s["g"]].sum() < a.num_gen]
    for t in TEACH:
        res[f"{t}_auc_mean_phi"] = auc(lambda s: mean(s[t]["phi"]))
        res[f"{t}_auc_sum_phi"] = auc(lambda s: float(np.sum(s[t]["phi"])))
        res[f"{t}_auc_pre_boxed"] = auc(lambda s: mean(s[t]["phi"][:s["cut"]]))
        res[f"{t}_auc_boxed_region"] = auc(lambda s: mean(s[t]["phi"][s["cut"]:]))
        res[f"{t}_auc_neg_kl"] = auc(lambda s: -mean(-s[t]["eph"]))          # E_pi[phi] = -KL(pi || p_T)
        allphi = np.concatenate([s[t]["phi"] for s in S])
        res[f"{t}_share_absphi_boxed_region"] = float(np.mean([np.abs(s[t]["phi"][s["cut"]:]).sum() / (np.abs(s[t]["phi"]).sum() + 1e-9) for s in S]))
        res[f"{t}_share_tokens_boxed_region"] = float(np.mean([(len(s["comp"]) - s["cut"]) / len(s["comp"]) for s in S]))
        res[f"{t}_frac_abs_phi_gt1"] = float((np.abs(allphi) > 1).mean())
        res[f"{t}_mean_phi"] = float(allphi.mean())
        # pooled token-level association with the advantage, mixed groups only
        ph_tok = np.concatenate([s[t]["phi"] - s[t]["eph"] for s in mixed]); A_tok = np.concatenate([np.full(len(s["comp"]), s["A"]) for s in mixed])
        res[f"{t}_corr_centred_phi_A_tokens"] = float(np.corrcoef(ph_tok, A_tok)[0, 1])
        pre = np.concatenate([(s[t]["phi"] - s[t]["eph"])[:s["cut"]] for s in mixed]); Apre = np.concatenate([np.full(s["cut"], s["A"]) for s in mixed])
        res[f"{t}_corr_centred_phi_A_pre_boxed"] = float(np.corrcoef(pre, Apre)[0, 1]) if len(pre) > 10 else None
        res[f"{t}_var_cv_token"] = float(np.var(ph_tok)); res[f"{t}_var_A_token"] = float(np.var(A_tok))
        # sequence level: corr(sum of centred phi, A) within mixed groups
        res[f"{t}_corr_seqsum_centred_phi_A"] = float(np.corrcoef([np.sum(s[t]["phi"] - s[t]["eph"]) for s in mixed], [s["A"] for s in mixed])[0, 1])
    # biggest |phi| tokens (full teacher), with context
    tops = []
    for s in random.sample(mixed, min(40, len(mixed))):
        ph = s["full"]["phi"]; k = int(np.argmax(np.abs(ph)))
        tops.append({"r": s["r"], "phi": round(float(ph[k]), 2), "pos": round(k / len(ph), 2), "in_boxed_region": k >= s["cut"],
                     "ctx": tok.decode(s["comp"][max(0, k - 12):k]) + " ⟦" + tok.decode([s["comp"][k]]) + "⟧"})
    res["top_tokens"] = tops
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "top_tokens"}), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
