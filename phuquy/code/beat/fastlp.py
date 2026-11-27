"""Faster, lower-memory per-token log-probs for the R01 loop (2026-11-26).

Profile of R01 on one A100 (2,048-token answers): forward+backward updates were 57% (GRPO) to 70% (Meta-OPSD) of each
rollout. Two causes, both fixed here without changing the numbers:

1. Micro-batches of a fixed 4 samples in random order: one long completion pads the whole batch. Here samples are sorted
   by length and packed under a token budget (padded length x count <= budget), so short samples share a batch.
2. log_softmax over the full vocabulary in fp32: 2,048 positions x 151,936 x 4 B = 1.25 GB per sample, kept for the
   backward pass. Here the LM head and the log-prob are computed per chunk of positions under activation checkpointing,
   so only the hidden states (2,048 x 2,048 x 2 B = 8 MB per sample) are kept; the logits are recomputed in backward.

Same maths as loop.logps_of: hidden -> lm_head (bf16) -> .float() -> log_softmax -> gather.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def token_batches(lengths, budget, max_items=None):
    """Index lists, longest first, each with max(length) * count <= budget (at least one item per batch)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and (m * (len(cur) + 1) > budget or (max_items and len(cur) >= max_items)):
            out.append(cur); cur, m = [], lengths[i]
        cur.append(i); cur_max = m
    if cur:
        out.append(cur)
    return out


def _head_logp(lm_head, h, tgt):
    logits = lm_head(h).float()
    return logits.gather(-1, tgt[:, None]).squeeze(-1) - torch.logsumexp(logits, -1)


def logps(model, pad_id, ctxs, comps, grad, budget, pos_chunk=1024):
    """Per-token log-prob (float32, 1-D) of each completion given its context, in the input order.

    model: the peft-wrapped causal LM. Left padding, explicit position ids (as loop.logps_of).
    """
    base = model.get_base_model()
    body, head = base.model, base.lm_head
    res = [None] * len(ctxs)
    lens = [len(c) + len(d) for c, d in zip(ctxs, comps)]
    for idx in token_batches(lens, budget):
        seqs = [ctxs[i] + comps[i] for i in idx]
        L = max(map(len, seqs))
        K = max(len(comps[i]) for i in idx) + 1
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s)
            att[j, L - len(s):] = 1
        ids, att = ids.cuda(), att.cuda()
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        with torch.set_grad_enabled(grad):
            hid = body(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state
            for j, i in enumerate(idx):
                n = len(comps[i])
                h = hid[j, L - 1 - n:L - 1]          # hidden state at position t predicts token t + 1
                tgt = ids[j, L - n:]
                parts = []
                for s in range(0, n, pos_chunk):
                    hs, ts = h[s:s + pos_chunk], tgt[s:s + pos_chunk]
                    parts.append(checkpoint(_head_logp, head, hs, ts, use_reentrant=False) if grad
                                 else _head_logp(head, hs, ts))
                res[i] = torch.cat(parts)
        del hid
    return res
