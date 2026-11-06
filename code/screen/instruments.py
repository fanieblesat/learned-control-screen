"""
instruments.py — the screening instruments of the JoH paper (DESIGN.md, Section 2) on one step interface.

A *problem adapter* exposes
    init(inst, rng)                       -> state            (state = (solution, cur, best))
    step(inst, state, theta, rng)         -> state            (one metaheuristic step, post-acceptance)
    score(state) -> cur  ;  value(state) -> best
All instruments below are written once against this interface. The EA adapters wrap the functions of
ea_core.py (a byte-identical copy of the EA 2026 suite) so that F and P reproduce the conference runs exactly
(same seeds, same random streams).

Instruments (Section 2 of DESIGN.md)
    fixed(theta)          F: theta at every step                      [EA run_fixed: rng = Random(seed)]
    probe(Theta)          P: one trial per value, keep the best trial [EA run_probe: master Random(10_000_019+seed)]
    null(theta*, K)       N: K trials of theta*, keep the best trial  [master Random(20_000_033+seed)]
    decoupled(Theta, m)   D_m: m trials per value, choose the best trial mean, ONE fresh step [Random(30_000_049+seed)]
    probe2(Theta)         P2: all ordered pairs, two consecutive steps each, keep the best    [Random(40_000_037+seed)]
    decoupled2(Theta, m)  D2_m: choose the pair with the best trial mean, execute it afresh [Random(50_000_017+seed)]
Budgets count STEPS (incumbent updates); a two-step decision consumes two steps.
"""
from __future__ import annotations
import random
from . import ea_core as ea

OFFSETS = dict(probe=10_000_019, null=20_000_033, decoupled=30_000_049, probe2=40_000_037, decoupled2=50_000_017)


# ------------------------------------------------------------------ problem adapters (EA settings)
class EAAdapter:
    """Wraps an EA setting dict (family pfsp | bap | control)."""

    def __init__(self, setting):
        self.st = setting; self.fam = setting["family"]

    def init(self, inst, rng):
        if self.fam == "pfsp":
            perm = ea.neh(inst); perm, cur = ea.local_search(inst, perm); return (perm, cur, cur)
        if self.fam == "bap":
            seqs = ea.bap_initial(inst); cur = ea.bap_cost(inst, seqs); return (seqs, cur, cur)
        x = rng.uniform(-30, 30); cur = ea.control_f(x); return (x, cur, cur)

    def step(self, inst, state, theta, rng):
        if self.fam == "pfsp":
            return ea.ils_step(inst, list(state[0]), state[1], state[2], self.st["mechanism"], theta, rng)
        if self.fam == "bap":
            return ea.lns_step(inst, [list(s) for s in state[0]], state[1], state[2], theta, rng)
        return ea.control_step(state[0], state[1], state[2], theta, rng)

    @staticmethod
    def score(state):
        return state[1]

    @staticmethod
    def value(state):
        return state[2]


# ------------------------------------------------------------------ instruments
def fixed(ad, inst, theta, T, seed, curve=None):
    """F(theta). Identical random stream to ea_core.run_fixed (one Random(seed) shared by init and all steps)."""
    rng = random.Random(seed)
    state = ad.init(inst, rng)
    for _ in range(T):
        state = ad.step(inst, state, theta, rng)
        if curve is not None: curve.append(ad.value(state))
    return ad.value(state)


def _best_of(ad, inst, state, thetas, master):
    """One trial per entry of `thetas` from `state`, independent streams; returns (winner_state, winner_index)."""
    trials = []
    for k, th in enumerate(thetas):
        r = random.Random(master.randrange(1 << 30))
        res = ad.step(inst, state, th, r)
        trials.append((ad.score(res), k, res))
    trials.sort(key=lambda t: t[0])          # ties -> lower index (stable on k), as in the EA probe
    return trials[0][2], trials[0][1]


def probe(ad, inst, Theta, T, seed, counts=None, curve=None, offset=OFFSETS["probe"]):
    """P. Identical to ea_core.run_probe for the EA adapters (same master stream, same trial streams, same tie rule)."""
    master = random.Random(offset + seed)
    state = ad.init(inst, master)
    for _ in range(T):
        state, k = _best_of(ad, inst, state, Theta, master)
        if counts is not None: counts[k] += 1
        if curve is not None: curve.append(ad.value(state))
    return ad.value(state)


def null(ad, inst, theta_star, K, T, seed, curve=None, offset=OFFSETS["null"]):
    """N: K independent trials of theta* per step; the best trial becomes the incumbent."""
    return probe(ad, inst, [theta_star] * K, T, seed, None, curve, offset)


def decoupled(ad, inst, Theta, m, T, seed, counts=None, curve=None, offset=OFFSETS["decoupled"]):
    """D_m: m trials per value from the incumbent (independent streams), choose the value with the best trial mean
    (ties -> lower index), then ONE fresh step with it on a new stream; the fresh step's result is the incumbent."""
    master = random.Random(offset + seed)
    state = ad.init(inst, master)
    for _ in range(T):
        means = []
        for k, th in enumerate(Theta):
            tot = 0.0
            for _ in range(m):
                r = random.Random(master.randrange(1 << 30))
                tot += ad.score(ad.step(inst, state, th, r))
            means.append(tot / m)
        lo = min(means); tied = [k for k, v in enumerate(means) if v <= lo + 1e-12]
        k = tied[0] if len(tied) == 1 else tied[master.randrange(len(tied))]   # ties broken uniformly at random
        state = ad.step(inst, state, Theta[k], random.Random(master.randrange(1 << 30)))
        if counts is not None: counts[k] += 1
        if curve is not None: curve.append(ad.value(state))
    return ad.value(state)


def _two_steps(ad, inst, state, a, b, master):
    s1 = ad.step(inst, state, a, random.Random(master.randrange(1 << 30)))
    return ad.step(inst, s1, b, random.Random(master.randrange(1 << 30)))


def probe2(ad, inst, Theta, T, seed, counts=None, offset=OFFSETS["probe2"]):
    """P2: every second step, all ordered pairs once (two consecutive steps, fresh streams); keep the best result.
    A decision consumes two steps; an odd last step is a one-step probe decision."""
    master = random.Random(offset + seed)
    state = ad.init(inst, master); t = 0
    pairs = [(a, b) for a in range(len(Theta)) for b in range(len(Theta))]
    while t < T:
        if T - t >= 2:
            best = None
            for (a, b) in pairs:
                res = _two_steps(ad, inst, state, Theta[a], Theta[b], master)
                key = (ad.score(res), a, b)
                if best is None or key < best[0]: best = (key, res)
            state = best[1]; t += 2
            if counts is not None: counts[(best[0][1], best[0][2])] += 1
        else:
            state, _ = _best_of(ad, inst, state, Theta, master); t += 1
    return ad.value(state)


def decoupled2(ad, inst, Theta, m, T, seed, counts=None, offset=OFFSETS["decoupled2"]):
    """D2_m: every second step, each ordered pair tried m times; the pair with the best trial mean is executed afresh."""
    master = random.Random(offset + seed)
    state = ad.init(inst, master); t = 0
    pairs = [(a, b) for a in range(len(Theta)) for b in range(len(Theta))]
    while t < T:
        if T - t >= 2:
            means = []
            for (a, b) in pairs:
                tot = sum(ad.score(_two_steps(ad, inst, state, Theta[a], Theta[b], master)) for _ in range(m))
                means.append((tot / m, a, b))
            lo = min(v for v, _, _ in means); tied = [(a, b) for v, a, b in means if v <= lo + 1e-12]
            a, b = tied[0] if len(tied) == 1 else tied[master.randrange(len(tied))]
            state = _two_steps(ad, inst, state, Theta[a], Theta[b], master); t += 2
            if counts is not None: counts[(a, b)] += 1
        else:
            sc = [ad.score(ad.step(inst, state, th, random.Random(master.randrange(1 << 30)))) for th in Theta]
            tied = [i for i, v in enumerate(sc) if v <= min(sc) + 1e-12]
            k = tied[0] if len(tied) == 1 else tied[master.randrange(len(tied))]
            state = ad.step(inst, state, Theta[k], random.Random(master.randrange(1 << 30))); t += 1
    return ad.value(state)
