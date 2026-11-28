"""Offline analysis of diag_fusion samples (seed 71, post hoc / exploratory). No model needed."""
import pickle, sys, collections
import numpy as np
rng = np.random.default_rng(0)
D = pickle.load(open(sys.argv[1], "rb")); CLS = D["CLS"]; S = D["samples"]
groups = collections.defaultdict(list)
for s in S:
    groups[(s["kind"] == "hard", s["q"])].append(s)
def env(key, g):
    if key[0]: return "hard32"
    k = int(sum(x["r"] for x in g)); return "mix1-2" if k <= 2 else "mix3-5" if k <= 5 else "mix6-7"
ENVS = ["mix1-2", "mix3-5", "mix6-7", "hard32"]
G = collections.defaultdict(list)
for key, g in groups.items(): G[env(key, g)].append(g)
print("questions per env", {e: len(G[e]) for e in ENVS})

def q_auc(g, score):
    pos = [score(s) for s in g if s["r"] == 1]; neg = [score(s) for s in g if s["r"] == 0]
    pos = [p for p in pos if p is not None]; neg = [n for n in neg if n is not None]
    if not pos or not neg: return None
    P, N = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(((P > N) + 0.5 * (P == N)).mean())
def env_auc(e, score, B=1000):
    v = np.array([a for a in (q_auc(g, score) for g in G[e]) if a is not None])
    if len(v) == 0: return (None, None, None, 0)
    bs = [rng.choice(v, len(v)).mean() for _ in range(B)]
    return (v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(v))
def fmt(t): return "   n/a   " if t[0] is None else f"{t[0]:.3f} [{t[1]:.2f},{t[2]:.2f}]"
def cmean(c, lo=0.0, hi=1.0):
    ci = CLS.index(c) if c else None
    def f(s):
        n = len(s["d"]); idx = np.arange(n); m = (idx >= lo * n) & (idx < hi * n)
        if ci is not None: m &= (s["cls"] == ci)
        return float(s["d"][m].mean()) if m.any() else None
    return f

print("\n1) Within-question AUC (correct vs wrong) of the teacher signal, by token class and difficulty environment [95% bootstrap CI]")
print(f"{'class':8s} " + " ".join(f"{e:>20s}" for e in ENVS))
for c in [None] + list(CLS) + ["nonprose_nobox"]:
    if c == "nonprose_nobox":
        keep = [CLS.index(x) for x in ("mnum", "msym", "num_out", "ws")]
        sc = lambda s: float(s["d"][np.isin(s["cls"], keep)].mean()) if np.isin(s["cls"], keep).any() else None
    else:
        sc = cmean(c)
    print(f"{(c or 'ALL'):8s} " + " ".join(f"{fmt(env_auc(e, sc)):>20s}" for e in ENVS))

print("\n2) Length as a confound: AUC of -length, and teacher signal with length partialled out (within question)")
neglen = lambda s: -len(s["d"])
print("   -length      " + " ".join(f"{fmt(env_auc(e, neglen)):>20s}" for e in ENVS))
for e in ENVS:
    xs, ys = [], []
    for g in G[e]:
        L = np.array([len(s["d"]) for s in g], float); M = np.array([s["d"].mean() for s in g])
        xs += list(np.log(L) - np.log(L).mean()); ys += list(M - M.mean())
    print(f"   corr(log length, mean d) within question, {e}: {np.corrcoef(xs, ys)[0,1]:+.3f}")
def resid_score(e):
    # per environment: regress mean d on log length (within question) and score by the residual
    xs, ys = [], []
    for g in G[e]:
        L = np.log([len(s["d"]) for s in g]); M = np.array([s["d"].mean() for s in g])
        xs += list(L - L.mean()); ys += list(M - M.mean())
    b = np.polyfit(xs, ys, 1)[0]
    return lambda s: float(s["d"].mean() - b * np.log(len(s["d"])))
print("   d | length   " + " ".join(f"{fmt(env_auc(e, resid_score(e))):>20s}" for e in ENVS))

print("\n3) Position: AUC of the teacher signal on the first / middle / last third of the answer (all tokens, then prose only)")
for c in (None, "prose", "msym"):
    for lo, hi, nm in ((0, 1/3, "first"), (1/3, 2/3, "middle"), (2/3, 1, "last")):
        print(f"   {(c or 'ALL'):6s} {nm:6s} " + " ".join(f"{fmt(env_auc(e, cmean(c, lo, hi))):>20s}" for e in ENVS))

print("\n4) The RL prior (class shares + position) per environment: within-question AUC of the 'noteacher' prior fitted on mixed-train")
beta = np.array(D["beta"]["noteacher"])
def prior(s):
    n = len(s["d"]); C = np.zeros((n, len(CLS))); C[np.arange(n), s["cls"]] = 1
    P = np.zeros((n, 3)); P[np.arange(n), np.minimum((np.arange(n) * 3) // n, 2)] = 1
    return float((np.concatenate([C, P], 1) @ beta).mean())
print("   prior        " + " ".join(f"{fmt(env_auc(e, prior)):>20s}" for e in ENVS))
print("   coefficients (class intercepts):", dict(zip(CLS, np.round(beta[:6], 4))))
for e in ENVS:
    sh = collections.defaultdict(list)
    for g in G[e]:
        for s in g:
            for i, c in enumerate(CLS): sh[(c, s["r"])].append((s["cls"] == i).mean())
    print(f"   {e}: share correct-vs-wrong  " + "  ".join(f"{c} {np.mean(sh[(c,1)]) - np.mean(sh[(c,0)]):+.3f}" for c in CLS))

print("\n5) Hard32 only: does the signal rank the 8 originally-failed samples vs the later correct ones only by the answer?")
H = G["hard32"]
for nm, sc in (("ALL", cmean(None)), ("box", cmean("box")), ("nonprose_nobox",
        lambda s: float(s["d"][np.isin(s["cls"], [1, 2, 3, 4])].mean()) if np.isin(s["cls"], [1, 2, 3, 4]).any() else None)):
    print(f"   {nm:15s} {fmt(env_auc('hard32', sc))}")
