"""Latent intervention pre-screen (no training): does y* act through a LATENT state of the model that can be transplanted?

Paired decoding. The student (plain prompt, no y*) samples its own solution token by token. A twin copy of the same base model
reads the SAME tokens with y* in context (HINT_TEMPLATE). At every step, the student's residual stream at the output of layer L,
at the current position, is replaced by the twin's:            do(h_L := h_L + P (h_L^twin - h_L))
P = identity (full patch) or the projection on the top-k directions of (h^twin - h) fitted on separate questions (low-rank patch).
Only the latent state is transplanted; no text of y* ever enters the student's context, and the tokens are the student's own.

Configs:
  none                     L = 0: the plain student                                            (anchor: plain)
  full@L                   full patch at layer L, L in --layers                                (mediation curve over depth)
  full@36                  last layer: student logits == twin logits                           (anchor: total effect of y*)
  rank{k}@L                low-rank patch, k in --ranks, at --rank-layers                      (is the effect carried by few directions?)
  placebo@L                full patch from a twin that reads y* of ANOTHER question            (frame effect vs content of y*)
Mediated share at a config = (acc - acc[none]) / (acc[full@last] - acc[none]).
Common random numbers: every config uses the same sampling seed per batch.

    PYTHONPATH=code python -m beat.diag_latent --questions 256 --eval-q 80 --out runs/diag_latent
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
    ap.add_argument("--questions", type=int, default=256)
    ap.add_argument("--eval-q", type=int, default=80)
    ap.add_argument("--fit-q", type=int, default=40)
    ap.add_argument("--samples", type=int, default=4, help="student samples per question per config")
    ap.add_argument("--hard-max", type=int, default=2)
    ap.add_argument("--layers", default="6,12,18,24,30")
    ap.add_argument("--ranks", default="8,64")
    ap.add_argument("--rank-layers", default="18,24")
    ap.add_argument("--placebo-layers", default="18")
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=80, help="student rows per batch (the twin doubles it)")
    ap.add_argument("--vllm-mem", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat.loop import HINT_TEMPLATE
    from train.common import BASE_MODEL, GRPO, configure_tokenizer
    from train.data import load_track
    from train.models import eos_ids
    from train.rewards import score_all

    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = configure_tokenizer(AutoTokenizer.from_pretrained(BASE_MODEL), "curated")
    stop_ids = sorted(set(eos_ids(tok, "curated")) | {tok.convert_tokens_to_ids("<|im_end|>")})
    pad_id = tok.pad_token_id
    T = GRPO["temperature"]

    rows = []
    for r in load_track("curated").itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.prompt]
        rows.append({"sys": msgs[0], "user": msgs[1]["content"], "q": r.extra_info["question"], "ref": r.target[0]["content"],
                     "gt": r.reward_model["ground_truth"]})
    Q = [rows[i] for i in np.random.permutation(len(rows))[:a.questions].tolist()]
    chat = lambda x, user: tok(tok.apply_chat_template([x["sys"], {"role": "user", "content": user}], tokenize=False,
                                                       add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    for k, x in enumerate(Q):
        x["pp"] = chat(x, x["user"])
        x["hp"] = chat(x, HINT_TEMPLATE.format(question=x["q"], reference=x["ref"]))
        x["plp"] = chat(x, HINT_TEMPLATE.format(question=x["q"], reference=Q[(k + 1) % len(Q)]["ref"]))
    Q = [x for x in Q if max(len(x["hp"]), len(x["plp"])) <= 1280]

    # ---- stage 1 (vLLM): hard set, and plain samples of the fit questions for the low-rank directions
    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=a.vllm_mem, max_model_len=4096, seed=a.seed, enable_sleep_mode=True)
    sp = SamplingParams(temperature=T, top_p=1.0, max_tokens=2048, stop_token_ids=stop_ids, n=8)
    o1 = llm.generate([{"prompt_token_ids": x["pp"]} for x in Q], sp, use_tqdm=False)
    r1 = score_all([c.text for o in o1 for c in o.outputs], [x["gt"] for x, o in zip(Q, o1) for _ in o.outputs], 8)
    k = 0
    for x, o in zip(Q, o1):
        x["s1"] = sum(r1[k:k + len(o.outputs)]); x["samp"] = [list(c.token_ids) for c in o.outputs]; k += len(o.outputs)
    llm.sleep(level=2); del llm
    torch.cuda.empty_cache()
    H = [x for x in Q if x["s1"] <= a.hard_max]
    random.shuffle(H)
    EV, FIT = H[:a.eval_q], H[a.eval_q:a.eval_q + a.fit_q]
    print(f"STAGE1 {time.time() - t0:.0f}s hard {len(H)}/{len(Q)} eval {len(EV)} fit {len(FIT)} "
          f"allfail_eval {sum(x['s1'] == 0 for x in EV)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    layers = model.model.layers
    NL = len(layers)
    PATCH = {"on": False, "B": 0, "U": None}

    def make_hook():
        def hook(mod, inp, outp):
            if not PATCH["on"]:
                return None
            h = outp[0] if isinstance(outp, tuple) else outp
            B = PATCH["B"]
            hs, ht = h[:B, -1], h[B:2 * B, -1]
            d = ht - hs
            if PATCH["U"] is not None:
                U = PATCH["U"]
                d = ((d.float() @ U) @ U.T).to(h.dtype)
            h[:B, -1] = hs + d
            return None
        return hook

    def left_pad(seqs):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s); mask[i, L - len(s):] = 1
        return ids.cuda(), mask.cuda()

    @torch.no_grad()
    def paired_decode(stu, twin, layer, U, seed):
        """stu, twin: lists of prompt ids (same length B). Returns list of generated token lists for the student rows."""
        B = len(stu)
        ids, mask = left_pad(stu + twin)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        handle = layers[layer - 1].register_forward_hook(make_hook()) if layer > 0 else None
        PATCH.update(on=layer > 0, B=B, U=U)
        g = torch.Generator(device="cuda").manual_seed(seed)
        cache = DynamicCache()
        try:
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True,
                           logits_to_keep=1).logits[:, -1]
            gen = [[] for _ in range(B)]
            done = torch.zeros(B, dtype=torch.bool, device="cuda")
            last = pos[:, -1]
            stop = torch.tensor(stop_ids, device="cuda")
            for t in range(a.max_new):
                p = torch.softmax(logits[:B].float() / T, -1)
                nxt = torch.multinomial(p, 1, generator=g)[:, 0]
                nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
                for i in torch.nonzero(~done).flatten().tolist():
                    gen[i].append(int(nxt[i]))
                done = done | torch.isin(nxt, stop)
                if bool(done.all()):
                    break
                inp = torch.cat([nxt, nxt])[:, None]
                mask = torch.cat([mask, torch.ones((2 * B, 1), dtype=mask.dtype, device="cuda")], 1)
                last = last + 1
                logits = model(input_ids=inp, attention_mask=mask, position_ids=last[:, None], past_key_values=cache,
                               use_cache=True).logits[:, -1]
        finally:
            PATCH["on"] = False
            if handle is not None:
                handle.remove()
        return gen

    # ---- low-rank directions: teacher-forced on plain samples of the FIT questions, both contexts
    @torch.no_grad()
    def diffs_at(layer, max_pos=60000):
        D = []
        for x in FIT:
            for s in x["samp"][:2]:
                hs = model(input_ids=torch.tensor([x["pp"] + s], device="cuda"), output_hidden_states=True,
                           logits_to_keep=1).hidden_states[layer][0, len(x["pp"]) - 1:len(x["pp"]) - 1 + len(s)]
                ht = model(input_ids=torch.tensor([x["hp"] + s], device="cuda"), output_hidden_states=True,
                           logits_to_keep=1).hidden_states[layer][0, len(x["hp"]) - 1:len(x["hp"]) - 1 + len(s)]
                D.append((ht - hs).float())
        D = torch.cat(D)
        if len(D) > max_pos:
            D = D[torch.randperm(len(D), device=D.device)[:max_pos]]
        return D

    ranks = [int(v) for v in a.ranks.split(",") if v]
    rank_layers = [int(v) for v in a.rank_layers.split(",") if v]
    BASIS, EXPL = {}, {}
    for L in rank_layers:
        D = diffs_at(L)
        _, S, Vh = torch.linalg.svd(D, full_matrices=False)
        ev = (S ** 2) / (S ** 2).sum()
        for kk in ranks:
            BASIS[(L, kk)] = Vh[:kk].T.contiguous()
            EXPL[f"rank{kk}@{L}"] = float(ev[:kk].sum())
        del D
    print(f"BASIS {time.time() - t0:.0f}s explained {json.dumps(EXPL)}", flush=True)

    # ---- configs
    cfgs = [("none", 0, None, "hp")]
    cfgs += [(f"full@{L}", L, None, "hp") for L in [int(v) for v in a.layers.split(",") if v]]
    cfgs += [(f"full@{NL}", NL, None, "hp")]
    cfgs += [(f"rank{kk}@{L}", L, (L, kk), "hp") for L in rank_layers for kk in ranks]
    cfgs += [(f"placebo@{L}", L, None, "plp") for L in [int(v) for v in a.placebo_layers.split(",") if v]]

    items = [(qi, j) for qi in range(len(EV)) for j in range(a.samples)]
    res = {"questions": len(Q), "hard": len(H), "eval_q": len(EV), "fit_q": len(FIT), "samples": a.samples,
           "num_layers": NL, "explained_var": EXPL, "configs": {}}
    per_q = {}
    for name, L, bkey, tw in cfgs:
        U = BASIS[bkey] if bkey else None
        texts, gts, lens, trunc, qidx = [], [], [], [], []
        for b0 in range(0, len(items), a.batch):
            chunk = items[b0:b0 + a.batch]
            gen = paired_decode([EV[qi]["pp"] for qi, _ in chunk], [EV[qi][tw] for qi, _ in chunk], L, U, a.seed + b0)
            for (qi, _), gtoks in zip(chunk, gen):
                texts.append(tok.decode(gtoks, skip_special_tokens=True)); gts.append(EV[qi]["gt"])
                lens.append(len(gtoks)); trunc.append(not gtoks or gtoks[-1] not in stop_ids); qidx.append(qi)
        rr = score_all(texts, gts, 8)
        acc_q = np.zeros(len(EV)); cnt = np.zeros(len(EV))
        for qi, r in zip(qidx, rr):
            acc_q[qi] += float(r); cnt[qi] += 1
        acc_q /= np.maximum(cnt, 1)
        per_q[name] = acc_q
        af = [i for i, x in enumerate(EV) if x["s1"] == 0]
        res["configs"][name] = {"layer": L, "acc": float(acc_q.mean()), "len": float(np.mean(lens)), "trunc": float(np.mean(trunc)),
                                "rescue_allfail": float(np.mean([acc_q[i] > 0 for i in af])) if af else None,
                                "example": texts[0][:400]}
        print(f"CFG {name} {time.time() - t0:.0f}s acc {acc_q.mean():.3f} len {np.mean(lens):.0f} trunc {np.mean(trunc):.3f}", flush=True)
        (out / "partial.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    rng = np.random.default_rng(a.seed)
    boot = lambda v: [float(np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(2000)], q)) for q in (5, 95)]
    base, top = per_q["none"], per_q[f"full@{NL}"]
    tot = float(top.mean() - base.mean())
    res["total_effect"] = tot
    for name in res["configs"]:
        d = per_q[name] - base
        res["configs"][name].update(delta=float(d.mean()), ci90=boot(d), mediated_share=float(d.mean() / tot) if tot else None)
    res["time"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT " + json.dumps({k: v for k, v in res.items()}), flush=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
