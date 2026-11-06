"""unit_checks.py — Step 3 of DESIGN.md (instrument part). Exact and distributional identities that must hold."""
import random, statistics as stt
from scipy import stats
from screen import ea_core as ea
from screen.instruments import EAAdapter, fixed, probe, null, decoupled, probe2, decoupled2

S = {s["label"]: s for s in ea.settings(False)}
ok = True
def report(name, cond, detail=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

# exact: N with K = 1 is the probe over {theta*} on the same stream
for lab in ["control", "bap-op-small"]:
    st = S[lab]; ad = EAAdapter(st); inst = None if st["family"] == "control" else ea.make_inst(st, 0)
    th = st["Theta"][1]
    a = [null(ad, inst, th, 1, st["T"], s) for s in range(20)]
    b = [probe(ad, inst, [th], st["T"], s, offset=20_000_033) for s in range(20)]
    report(f"N(K=1) == P({{theta}}) exactly [{lab}]", a == b)

# distributional: D_1 over {theta}, N(K=1), P2 over {theta} all have the law of F(theta) (control, 400 seeds)
st = S["control"]; ad = EAAdapter(st); th = 2.0; n = 400
F = [fixed(ad, None, th, st["T"], 1_000_000 + s) for s in range(n)]
for name, xs in [("D_1({theta})", [decoupled(ad, None, [th], 1, st["T"], s) for s in range(n)]),
                 ("N(K=1)", [null(ad, None, th, 1, st["T"], s) for s in range(n)]),
                 ("P2({theta})", [probe2(ad, None, [th], st["T"], s) for s in range(n)]),
                 ("D2_1({theta})", [decoupled2(ad, None, [th], 1, st["T"], s) for s in range(n)])]:
    p = stats.mannwhitneyu(xs, F).pvalue
    report(f"{name} ~ F(theta) in law [control]", p > 0.01, f"(Mann-Whitney p = {p:.3f}; means {stt.mean(xs):.3f} vs {stt.mean(F):.3f})")

# order: on the positive control the probe beats the null probe (headroom exists by construction)
P = [probe(ad, None, st["Theta"], st["T"], s) for s in range(200)]
N = [null(ad, None, 8.0, 4, st["T"], s) for s in range(200)]
D = [decoupled(ad, None, st["Theta"], 1, st["T"], s) for s in range(200)]
F8 = [fixed(ad, None, 8.0, st["T"], 2_000_000 + s) for s in range(200)]
report("control: mean P < mean N", stt.mean(P) < stt.mean(N), f"({stt.mean(P):.3f} vs {stt.mean(N):.3f})")
report("control: mean D_1 < mean F(8.0)", stt.mean(D) < stt.mean(F8), f"({stt.mean(D):.3f} vs {stt.mean(F8):.3f})")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
