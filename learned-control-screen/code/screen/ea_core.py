"""
EA 2026 camera-ready experimental suite
=======================================
"When Does Learned Control Help? An Oracle Analysis of Adaptive Parameter
Control for Metaheuristics on Scheduling Problems"
(Niebles-Atencio & Voss, EA 2026, Nice)

Regenerates ALL numbers for the camera-ready with the replication protocol
requested by the reviewers, and produces every figure and LaTeX table row.

DESIGN DECISIONS (these define the probe; stated identically in the paper):
  D1  Probe: at each iteration, each candidate theta gets exactly ONE trial
      from the current incumbent, with INDEPENDENT random draws per trial.
  D2  The winning trial's resulting incumbent IS the next incumbent
      (the step is not re-executed).
  D3  The scored quantity is the post-acceptance incumbent (for mechanisms
      with an acceptance rule, the candidate is scored after accept/reject).
  D4  T = 0 denotes strictly greedy acceptance (exp() never evaluated).
  D5  Best-fixed baseline is selected on an INDEPENDENT seed split:
      seeds 0..SEL-1 select the best fixed theta per (mechanism, scale,
      instance); seeds SEL..SEL+EVAL-1 are used for all comparisons.
  D6  All instances, runs, and trainings are deterministically seeded.

USAGE
  python ea2026_suite.py all            # full study (overnight; prints ETA)
  python ea2026_suite.py all --fast     # smoke test, ~2-4 min
  python ea2026_suite.py stats          # recompute stats/tables from CSVs
  python ea2026_suite.py figures        # regenerate all figures from CSVs
  python ea2026_suite.py controller     # tabular-Q controller on BAP-medium

OUTPUT
  results/*.csv        per-run records (regenerable exactly)
  figures/*.pdf        camera-ready figures
  printed LaTeX rows   transcribe into Tables 1-2 and Sec. 4.4
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

try:
    from scipy import stats as sps
except ImportError:
    sps = None
    print("WARNING: scipy not found -> p-values will be reported as 'n/a'. "
          "Install with: pip install scipy --break-system-packages")

SEL_SEEDS = 5      # seeds used only to select the best fixed theta (D5)
EVAL_SEEDS = 25    # seeds used for all reported comparisons
RESULTS = "results"
FIGURES = "figures"


# ============================================================ PFSP ==========
def pfsp_instance(n_jobs, n_mach, seed):
    rng = np.random.RandomState(seed)
    return rng.randint(1, 100, size=(n_jobs, n_mach)).astype(float)


def makespan(P, perm):
    """O(n m) makespan of permutation perm on processing matrix P."""
    C = np.zeros(P.shape[1])
    for j in perm:
        C[0] += P[j, 0]
        for m in range(1, P.shape[1]):
            C[m] = max(C[m], C[m - 1]) + P[j, m]
    return C[-1]


def _heads(P, perm):
    n, m = len(perm), P.shape[1]
    E = np.zeros((n + 1, m))
    for i, j in enumerate(perm):
        E[i + 1, 0] = E[i, 0] + P[j, 0]
        for k in range(1, m):
            E[i + 1, k] = max(E[i + 1, k - 1], E[i, k]) + P[j, k]
    return E


def _tails(P, perm):
    n, m = len(perm), P.shape[1]
    Q = np.zeros((n + 1, m))
    for i in range(n - 1, -1, -1):
        j = perm[i]
        Q[i, m - 1] = Q[i + 1, m - 1] + P[j, m - 1]
        for k in range(m - 2, -1, -1):
            Q[i, k] = max(Q[i, k + 1], Q[i + 1, k]) + P[j, k]
    return Q


def best_insertion(P, perm, job):
    """Taillard acceleration: best position & makespan for inserting job."""
    n, m = len(perm), P.shape[1]
    E, Q = _heads(P, perm), _tails(P, perm)
    best_c, best_pos = math.inf, 0
    for pos in range(n + 1):
        f = np.zeros(m)
        f[0] = E[pos, 0] + P[job, 0]
        for k in range(1, m):
            f[k] = max(f[k - 1], E[pos, k]) + P[job, k]
        c = np.max(f + Q[pos]) if pos < n + 1 else f[-1]
        if c < best_c - 1e-9:
            best_c, best_pos = c, pos
    return best_pos, best_c


def neh(P):
    order = np.argsort(-P.sum(axis=1))
    perm = [int(order[0])]
    for job in order[1:]:
        pos, _ = best_insertion(P, perm, int(job))
        perm.insert(pos, int(job))
    return perm


def local_search(P, perm):
    """First-improvement insertion neighbourhood, to local optimality."""
    perm = list(perm)
    cur = makespan(P, perm)
    improved = True
    while improved:
        improved = False
        for j in list(perm):
            rest = [x for x in perm if x != j]
            pos, c = best_insertion(P, rest, j)
            if c < cur - 1e-9:
                rest.insert(pos, j)
                perm, cur, improved = rest, c, True
                break
    return perm, cur


def ils_step(P, perm, cur, best, mechanism, theta, rng):
    """One ILS iteration: perturb -> local search -> acceptance (D3, D4).

    mechanism 'perturb': theta = # jobs removed+reinserted; acceptance greedy.
    mechanism 'accept':  theta = temperature T; perturbation fixed at 4.
    Returns (perm, cur, best) post-acceptance.
    """
    k = theta if mechanism == "perturb" else 4
    cand = list(perm)
    jobs = rng.sample(cand, min(k, len(cand)))
    for j in jobs:
        cand.remove(j)
    for j in jobs:
        pos, _ = best_insertion(P, cand, j)
        cand.insert(pos, j)
    cand, c = local_search(P, cand)
    best = min(best, c)
    if mechanism == "accept":
        T = theta
        if c <= cur + 1e-9:
            return cand, c, best
        if T <= 0:                                   # D4: strictly greedy
            return perm, cur, best
        c_ref = cur
        if rng.random() < math.exp(-(c - cur) / (T * c_ref)):
            return cand, c, best
        return perm, cur, best
    # perturbation mechanism: greedy acceptance
    if c <= cur + 1e-9:
        return cand, c, best
    return perm, cur, best


# ============================================================ BAP ===========
def bap_instance(n_vessels, n_berths, seed):
    rng = np.random.RandomState(seed)
    cls = rng.choice(3, size=n_vessels, p=[0.5, 0.35, 0.15])
    h = np.where(cls == 0, rng.uniform(4, 8, n_vessels),
                 np.where(cls == 1, rng.uniform(10, 18, n_vessels),
                          rng.uniform(20, 35, n_vessels)))
    horizon = h.sum() / n_berths * 0.55        # ~75-80% contention under FCFS
    a = rng.uniform(0, horizon, n_vessels)
    w = rng.choice([3, 2, 1], size=n_vessels, p=[0.2, 0.3, 0.5]).astype(float)
    return {"a": a, "h": h, "w": w, "n": n_vessels, "m": n_berths}


def bap_cost(inst, seqs):
    """Total priority-weighted turnaround of per-berth sequences."""
    total = 0.0
    for seq in seqs:
        t = 0.0
        for v in seq:
            s = max(t, inst["a"][v])
            t = s + inst["h"][v]
            total += inst["w"][v] * (t - inst["a"][v])
    return total


def bap_repair(inst, seqs, removed):
    """Greedy insertion: heaviest (w*h) first, best (berth, position)."""
    for v in sorted(removed, key=lambda x: -inst["w"][x] * inst["h"][x]):
        best = (math.inf, 0, 0)
        for b in range(inst["m"]):
            for pos in range(len(seqs[b]) + 1):
                seqs[b].insert(pos, v)
                c = bap_cost(inst, seqs)
                seqs[b].pop(pos)
                if c < best[0]:
                    best = (c, b, pos)
        seqs[best[1]].insert(best[2], v)
    return seqs


def bap_initial(inst):
    order = np.argsort(inst["a"])
    seqs = [[] for _ in range(inst["m"])]
    free = np.zeros(inst["m"])
    for v in order:
        b = int(np.argmin(np.maximum(free, inst["a"][v])))
        seqs[b].append(int(v))
        free[b] = max(free[b], inst["a"][v]) + inst["h"][v]
    return seqs


def _flat(seqs):
    return [v for s in seqs for v in s]


def bap_destroy(inst, seqs, op, q, rng):
    vessels = _flat(seqs)
    if op == 0:                                          # random
        rem = rng.sample(vessels, q)
    elif op == 1:                                        # worst-cost
        contrib = {}
        for b, seq in enumerate(seqs):
            t = 0.0
            for v in seq:
                s = max(t, inst["a"][v]); t = s + inst["h"][v]
                contrib[v] = inst["w"][v] * (t - inst["a"][v])
        rem = sorted(vessels, key=lambda v: -contrib[v])[:q]
    elif op == 2:                                        # Shaw relatedness
        seedv = rng.choice(vessels)
        def rel(v):
            return (abs(inst["a"][v] - inst["a"][seedv]) / 10.0
                    + abs(inst["h"][v] - inst["h"][seedv]) / 5.0)
        rem = sorted(vessels, key=rel)[:q]
    elif op == 3:                                        # cluster (one berth)
        b = rng.randrange(inst["m"])
        pool = list(seqs[b]) or vessels
        rem = pool[:q] if len(pool) >= q else \
            pool + rng.sample([v for v in vessels if v not in pool],
                              q - len(pool))
    else:                                                # time-window
        t0 = rng.uniform(0, max(1e-9, inst["a"].max()))
        win = sorted(vessels, key=lambda v: abs(inst["a"][v] - t0))
        rem = win[:q]
    new = [[v for v in s if v not in set(rem)] for s in seqs]
    return new, list(rem)


def lns_step(inst, seqs, cur, best, theta, rng):
    """One LNS iteration with operator theta in {0..4}; greedy acceptance."""
    q = max(2, int(0.2 * inst["n"]))
    cand, removed = bap_destroy(inst, [list(s) for s in seqs], theta, q, rng)
    cand = bap_repair(inst, cand, removed)
    c = bap_cost(inst, cand)
    best = min(best, c)
    if c <= cur + 1e-9:
        return cand, c, best
    return seqs, cur, best


# ================================================== positive control ========
def control_f(x):
    return x * x / 100.0 + 10.0 * math.sin(x) ** 2


def control_step(x, cur, best, theta, rng):
    cand = x + rng.gauss(0, theta)
    c = control_f(cand)
    best = min(best, c)
    if c <= cur:
        return cand, c, best
    return x, cur, best


# ================================================== generic runners =========
def run_fixed(setting, inst, theta, T, seed, log_curve=None):
    rng = random.Random(seed)
    if setting["family"] == "pfsp":
        P = inst
        perm = neh(P)
        perm, cur = local_search(P, perm)
        best = cur
        for i in range(T):
            perm, cur, best = ils_step(P, perm, cur, best,
                                       setting["mechanism"], theta, rng)
            if log_curve is not None:
                log_curve.append(best)
        return best
    if setting["family"] == "bap":
        seqs = bap_initial(inst)
        cur = bap_cost(inst, seqs)
        best = cur
        for i in range(T):
            seqs, cur, best = lns_step(inst, seqs, cur, best, theta, rng)
            if log_curve is not None:
                log_curve.append(best)
        return best
    # control
    x = rng.uniform(-30, 30)
    cur = control_f(x)
    best = cur
    for i in range(T):
        x, cur, best = control_step(x, cur, best, theta, rng)
        if log_curve is not None:
            log_curve.append(best)
    return best


def run_probe(setting, inst, Theta, T, seed, sel_counts=None, log_curve=None):
    """Greedy one-step probe per D1-D3: K independent trials, keep winner."""
    rng = random.Random(10_000_019 + seed)
    fam = setting["family"]
    if fam == "pfsp":
        P = inst
        perm = neh(P); perm, cur = local_search(P, perm); best = cur
        state = (perm, cur, best)
    elif fam == "bap":
        seqs = bap_initial(inst); cur = bap_cost(inst, seqs)
        state = (seqs, cur, cur)
    else:
        x = rng.uniform(-30, 30); cur = control_f(x)
        state = (x, cur, cur)
    for i in range(T):
        trials = []
        for k, th in enumerate(Theta):
            r = random.Random(rng.randrange(1 << 30))   # D1: independent
            if fam == "pfsp":
                res = ils_step(P, list(state[0]), state[1], state[2],
                               setting["mechanism"], th, r)
            elif fam == "bap":
                res = lns_step(inst, [list(s) for s in state[0]],
                               state[1], state[2], th, r)
            else:
                res = control_step(state[0], state[1], state[2], th, r)
            trials.append((res[1], k, res))             # D3: post-acceptance
        trials.sort(key=lambda t: t[0])
        _, kstar, state = trials[0]                     # D2: winner kept
        if sel_counts is not None:
            sel_counts[kstar] += 1
        if log_curve is not None:
            log_curve.append(state[2])
    return state[2]


# ================================================== study definition ========
def settings(fast=False):
    pf_scales = [("small", 20, 5, 150), ("medium", 50, 10, 120),
                 ("large", 80, 10, 80)]
    bap_scales = [("small", 10, 3, 300), ("medium", 20, 4, 250),
                  ("large", 40, 8, 150)]
    if fast:
        pf_scales = [("small", 12, 5, 25)]
        bap_scales = [("small", 8, 3, 40), ("medium", 12, 3, 40)]
    out = []
    for name, n, m, T in pf_scales:
        for mech, Theta in [("perturb", [2, 4, 6, 8, 12]),
                            ("accept", [0.0, 0.002, 0.005, 0.01, 0.02])]:
            out.append(dict(family="pfsp", mechanism=mech, scale=name,
                            n=n, m=m, T=T, Theta=Theta,
                            label=f"pfsp-{mech}-{name}"))
    for name, n, m, T in bap_scales:
        out.append(dict(family="bap", mechanism="operator", scale=name,
                        n=n, m=m, T=T, Theta=[0, 1, 2, 3, 4],
                        label=f"bap-op-{name}"))
    out.append(dict(family="control", mechanism="step", scale="-",
                    n=0, m=0, T=100, Theta=[0.5, 2.0, 8.0, 20.0],
                    label="control"))
    return out


def make_inst(st, idx):
    seed = 100_000 + 1000 * idx
    if st["family"] == "pfsp":
        return pfsp_instance(st["n"], st["m"], seed + st["n"])
    if st["family"] == "bap":
        return bap_instance(st["n"], st["m"], seed + st["n"])
    return None


def run_study(fast=False):
    os.makedirs(RESULTS, exist_ok=True)
    n_inst = 2 if fast else 6
    sel_s = 2 if fast else SEL_SEEDS
    ev_s = 3 if fast else EVAL_SEEDS
    n_ctrl = 5 if fast else 30
    rows, t0 = [], time.time()
    for st in settings(fast):
        part_path = f"{RESULTS}/runs_{st['label']}.csv"
        if (not fast) and os.path.exists(part_path):
            print(f"  {st['label']:<22} RESUME: found {part_path}, skipping")
            continue
        srows = []
        insts = ([make_inst(st, i) for i in range(n_inst)]
                 if st["family"] != "control" else [None] * n_ctrl)
        sel_counts = defaultdict(int)
        for i, inst in enumerate(insts):
            seeds_all = range(sel_s + ev_s)
            for th_idx, th in enumerate(st["Theta"]):
                for s in seeds_all:
                    v = run_fixed(st, inst, th, st["T"], 777 + 97 * i + s)
                    srows.append(dict(label=st["label"], instance=i,
                                     kind="fixed", theta_idx=th_idx,
                                     seed=s, split=("sel" if s < sel_s
                                                    else "eval"), value=v))
            for s in range(sel_s, sel_s + ev_s):
                curve = [] if i == 0 and s == sel_s else None
                v = run_probe(st, inst, st["Theta"], st["T"],
                              777 + 97 * i + s, sel_counts, curve)
                srows.append(dict(label=st["label"], instance=i, kind="probe",
                                 theta_idx=-1, seed=s, split="eval", value=v))
                if curve:
                    with open(f"{RESULTS}/curve_{st['label']}.csv", "w") as f:
                        f.write("iter,probe_best\n")
                        f.writelines(f"{j},{b}\n" for j, b in enumerate(curve))
        with open(f"{RESULTS}/selection_{st['label']}.csv", "w") as f:
            f.write("theta_idx,count\n")
            for k in range(len(st["Theta"])):
                f.write(f"{k},{sel_counts[k]}\n")
        with open(part_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
            w.writeheader(); w.writerows(srows)
        rows += srows
        print(f"  {st['label']:<22} done  ({time.time()-t0:6.0f}s)  "
              f"[saved {part_path}]")
    print(f"All settings complete; per-setting files in {RESULTS}/")
    compute_stats(fast)


# ================================================== statistics ==============
def load_runs():
    import glob
    import pandas as pd
    parts = sorted(glob.glob(f"{RESULTS}/runs_*.csv"))
    if parts:
        return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    return pd.read_csv(f"{RESULTS}/runs.csv")


def compute_stats(fast=False):
    df = load_runs()
    print("\n--- LaTeX table rows (Tables 1-2) ---")
    print("label & spread% & varies & probe_gain% & p & sigma% \\\\")
    for label in df.label.unique():
        d = df[df.label == label]
        fx = d[d.kind == "fixed"]
        # best fixed per instance chosen on SELECTION split only (D5)
        sel = (fx[fx.split == "sel"]
               .groupby(["instance", "theta_idx"]).value.mean().reset_index())
        best_th = sel.loc[sel.groupby("instance").value.idxmin()] \
                     .set_index("instance").theta_idx.to_dict()
        varies = "yes" if len(set(best_th.values())) > 1 else "no"
        ev = fx[fx.split == "eval"]
        means = ev.groupby(["instance", "theta_idx"]).value.mean().unstack()
        spread = ((means.max(axis=1) - means.min(axis=1))
                  / means.min(axis=1) * 100).mean()
        probe = d[d.kind == "probe"].set_index(["instance", "seed"]).value
        gains, pairs_p, pairs_b = [], [], []
        for inst in means.index:
            bth = best_th.get(inst, means.loc[inst].idxmin())
            bf = ev[(ev.instance == inst) & (ev.theta_idx == bth)] \
                .set_index("seed").value
            pr = probe.loc[inst]
            common = bf.index.intersection(pr.index)
            pairs_b += list(bf.loc[common]); pairs_p += list(pr.loc[common])
            gains.append((bf.loc[common].mean() - pr.loc[common].mean())
                         / bf.loc[common].mean() * 100)
        gain = float(np.mean(gains))   # positive = probe IMPROVES on best fixed
        sigma = float((ev.groupby(["instance", "theta_idx"]).value.std()
                       / ev.groupby(["instance", "theta_idx"]).value.mean())
                      .mean() * 100)
        if sps is not None and len(pairs_b) > 5 and \
                any(abs(a - b) > 1e-12 for a, b in zip(pairs_b, pairs_p)):
            p = sps.wilcoxon(pairs_p, pairs_b).pvalue
            pstr = f"{p:.1e}"
        else:
            pstr = "n/a"
        print(f"{label} & {spread:.2f} & {varies} & {gain:+.2f} & "
              f"{pstr} & {sigma:.2f} \\\\")
    print("\n(sign convention: probe_gain > 0 means the probe IMPROVES on "
          "the best fixed parameter, matching the paper's tables; negative "
          "values mean the probe loses to best-fixed over the trajectory)")


# ================================================== figures =================
def make_figures():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "figure.dpi": 150})
    os.makedirs(FIGURES, exist_ok=True)
    df = load_runs()

    # recompute per-label probe gain (eval split, D5 baseline)
    def label_gain(d):
        fx = d[d.kind == "fixed"]
        sel = (fx[fx.split == "sel"]
               .groupby(["instance", "theta_idx"]).value.mean().reset_index())
        best_th = sel.loc[sel.groupby("instance").value.idxmin()] \
                     .set_index("instance").theta_idx.to_dict()
        ev = fx[fx.split == "eval"]
        means = ev.groupby(["instance", "theta_idx"]).value.mean().unstack()
        probe = d[d.kind == "probe"].groupby("instance").value.mean()
        gains = []
        for inst in means.index:
            bth = best_th.get(inst, means.loc[inst].idxmin())
            bf = means.loc[inst, bth]
            gains.append((probe.loc[inst] - bf) / bf * 100)
        return -float(np.mean(gains))     # + = probe improves (minimization)

    labels = [l for l in df.label.unique()]
    gains = {l: label_gain(df[df.label == l]) for l in labels}

    # Fig 1: headroom spectrum
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    order = sorted(labels, key=lambda l: gains[l])
    colors = ["#1f4e79" if l.startswith("pfsp") else
              "#2e7d32" if l.startswith("bap") else "#e08214" for l in order]
    ax.barh(range(len(order)), [gains[l] for l in order], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.axvline(1.0, ls="--", color="red", lw=0.8)
    ax.set_xlabel("Probe gain over best fixed parameter (%)")
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/fig_headroom_spectrum.pdf"); plt.close()

    # Fig: selection distributions (PFSP + BAP)
    sel_labels = [l for l in labels if not l.startswith("control")]
    ncol = 3
    nrow = math.ceil(len(sel_labels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6, 1.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, l in zip(axes, sel_labels):
        s = pd.read_csv(f"{RESULTS}/selection_{l}.csv")
        share = s["count"] / s["count"].sum() * 100
        ax.bar(s.theta_idx, share, color="#1f4e79")
        ax.axhline(100 / len(s), ls="--", color="gray", lw=0.8)
        ax.set_title(l, fontsize=7); ax.set_ylim(0, 100)
        ax.tick_params(labelsize=6)
    for ax in axes[len(sel_labels):]:
        ax.axis("off")
    fig.supylabel("Selection share (%)", fontsize=8)
    fig.supxlabel("Candidate index", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/fig_selection_distribution.pdf"); plt.close()

    # Fig: probe best-so-far curves (oracle over time)
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for l in labels:
        path = f"{RESULTS}/curve_{l}.csv"
        if os.path.exists(path):
            c = pd.read_csv(path)
            ax.plot(c["iter"], c.probe_best / c.probe_best.iloc[0],
                    label=l, lw=1)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Probe best (normalized)")
    ax.legend(fontsize=5, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/fig_oracle_over_time.pdf"); plt.close()
    print(f"Figures written to {FIGURES}/")


# ================================================== controller ==============
def train_controller(fast=False):
    """Tabular Q-learning operator selector on BAP-medium (R3 Sec 2.6 fix).

    State: (stagnation bucket 0/1/2, last-op-improved bit) -> 6 states.
    Action: operator 0..4. Reward: relative incumbent improvement.
    Trained on training seeds, evaluated greedily on the eval protocol,
    compared against best-fixed (D5 baseline) and the probe.
    """
    st = dict(family="bap", mechanism="operator", scale="medium",
              n=(12 if fast else 20), m=(3 if fast else 4),
              T=(40 if fast else 250), Theta=[0, 1, 2, 3, 4])
    n_inst = 2 if fast else 6
    episodes = 30 if fast else 400
    insts = [make_inst(st, i) for i in range(n_inst)]
    Q = defaultdict(float)
    alpha, gamma, eps = 0.2, 0.9, 0.2
    rng = random.Random(7)

    def state_of(stag, improved):
        return (0 if stag == 0 else 1 if stag < 10 else 2, int(improved))

    for ep in range(episodes):
        inst = insts[ep % n_inst]
        seqs = bap_initial(inst)
        cur = bap_cost(inst, seqs); best = cur
        stag, improved = 0, 0
        for i in range(st["T"]):
            s = state_of(stag, improved)
            a = (rng.randrange(5) if rng.random() < eps else
                 max(range(5), key=lambda x: Q[(s, x)]))
            prev_best = best
            seqs, cur, best = lns_step(inst, seqs, cur, best, a,
                                       random.Random(rng.randrange(1 << 30)))
            r = (prev_best - best) / prev_best
            improved = int(best < prev_best - 1e-9)
            stag = 0 if improved else stag + 1
            s2 = state_of(stag, improved)
            Q[(s, a)] += alpha * (r + gamma *
                                  max(Q[(s2, x)] for x in range(5))
                                  - Q[(s, a)])
    # greedy evaluation on eval seeds
    vals = []
    for i, inst in enumerate(insts):
        for sd in range(SEL_SEEDS, SEL_SEEDS + (3 if fast else EVAL_SEEDS)):
            r2 = random.Random(777 + 97 * i + sd)
            seqs = bap_initial(inst)
            cur = bap_cost(inst, seqs); best = cur
            stag, improved = 0, 0
            for _ in range(st["T"]):
                s = state_of(stag, improved)
                a = max(range(5), key=lambda x: Q[(s, x)])
                pb = best
                seqs, cur, best = lns_step(inst, seqs, cur, best, a,
                                           random.Random(r2.randrange(1 << 30)))
                improved = int(best < pb - 1e-9)
                stag = 0 if improved else stag + 1
            vals.append(dict(instance=i, seed=sd, value=best))
    with open(f"{RESULTS}/controller_bap_medium.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "seed", "value"])
        w.writeheader(); w.writerows(vals)
    print(f"Saved {RESULTS}/controller_bap_medium.csv "
          f"({len(vals)} evals). Compare against bap-op-medium rows in "
          f"stats output: controller mean per instance vs best-fixed vs probe.")
    m = np.mean([v["value"] for v in vals])
    print(f"Controller mean objective: {m:.1f}")
    # automatic gap vs best-fixed (D5) and probe, if study CSV exists
    try:
        import pandas as pd
        df = load_runs()
        d = df[df.label == "bap-op-medium"]
        fx = d[d.kind == "fixed"]
        sel = (fx[fx.split == "sel"]
               .groupby(["instance", "theta_idx"]).value.mean().reset_index())
        best_th = sel.loc[sel.groupby("instance").value.idxmin()] \
                     .set_index("instance").theta_idx.to_dict()
        ev = fx[fx.split == "eval"]
        cg, pg = [], []
        cdf = pd.DataFrame(vals)
        for i in best_th:
            bf = ev[(ev.instance == i) &
                    (ev.theta_idx == best_th[i])].value.mean()
            ct = cdf[cdf.instance == i].value.mean()
            pr = d[(d.kind == "probe") & (d.instance == i)].value.mean()
            cg.append((bf - ct) / bf * 100)
            pg.append((bf - pr) / bf * 100)
        print(f"Controller gain vs best-fixed: {np.mean(cg):+.2f}%  "
              f"(probe gain on same protocol: {np.mean(pg):+.2f}%)  "
              f"-> transcribe into Sec. 4.4 [[CONTROLLER]] row")
    except Exception as e:
        print(f"(gap computation skipped: {e})")


# ================================================== main ====================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "stats", "figures", "controller"])
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast:
        RESULTS, FIGURES = "results_fast", "figures_fast"
    if args.cmd == "all":
        run_study(args.fast)
        make_figures()
        train_controller(args.fast)
    elif args.cmd == "stats":
        compute_stats()
    elif args.cmd == "figures":
        make_figures()
    else:
        train_controller(args.fast)
