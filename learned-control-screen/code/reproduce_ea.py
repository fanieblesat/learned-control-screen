"""
reproduce_ea.py — Step 1 of DESIGN.md: the refactored instruments must reproduce the EA 2026 runs EXACTLY
(same seeds) before any new code is trusted. Compares every run of the chosen EA settings, value by value,
with the EA records in ea_reference/results/runs_<label>.csv.

Usage: python reproduce_ea.py [labels...] [--instances i,j]
  default: control and bap-op-small in full; pfsp-perturb-small and pfsp-accept-small on instance 0.
"""
import argparse, csv, sys, time
from screen import ea_core as ea
from screen.instruments import EAAdapter, fixed, probe

REF = "ea_reference/results"


def ea_settings():
    return {s["label"]: s for s in ea.settings(False)}


def check(label, instances=None):
    st = ea_settings()[label]; ad = EAAdapter(st)
    rows = list(csv.DictReader(open(f"{REF}/runs_{label}.csv")))
    n_inst = 30 if st["family"] == "control" else 6
    insts = [None] * n_inst if st["family"] == "control" else [ea.make_inst(st, i) for i in range(n_inst)]
    t0 = time.time(); n = bad = 0; worst = 0.0
    for r in rows:
        i = int(r["instance"])
        if instances is not None and i not in instances:
            continue
        s = int(r["seed"]); seed = 777 + 97 * i + s
        if r["kind"] == "fixed":
            v = fixed(ad, insts[i], st["Theta"][int(r["theta_idx"])], st["T"], seed)
        else:
            v = probe(ad, insts[i], st["Theta"], st["T"], seed)
        ref = float(r["value"]); n += 1
        if v != ref:
            bad += 1; worst = max(worst, abs(v - ref))
    print(f"{label:<22} runs checked {n:5d}  mismatches {bad}  max |diff| {worst:.3g}  ({time.time()-t0:.0f} s)", flush=True)
    return bad == 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("labels", nargs="*"); ap.add_argument("--instances", default=None)
    a = ap.parse_args()
    inst = set(int(x) for x in a.instances.split(",")) if a.instances else None
    plan = [(l, inst) for l in a.labels] if a.labels else \
        [("control", None), ("bap-op-small", None), ("pfsp-perturb-small", {0}), ("pfsp-accept-small", {0})]
    ok = all(check(l, i) for l, i in plan)
    print("REPRODUCED EXACTLY" if ok else "MISMATCH: do not use the refactored code until resolved")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
