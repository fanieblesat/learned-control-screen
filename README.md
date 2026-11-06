# Screening adaptive parameter control before learning

Code and protocol of the Journal of Heuristics extension of Niebles-Atencio & Voss (EA 2026, LNCS, to appear).

- `DESIGN.md` — the study protocol. Frozen from the commit that adds it; later changes only through its Section 11.
- `code/screen/ea_core.py` — byte-identical copy of the EA 2026 suite (`ea2026_suite.py`).
- `code/screen/instruments.py` — fixed run, probe, null probe, decoupled probe, two-step probes on one step interface.
- `code/reproduce_ea.py` — exact reproduction of the EA runs (`python reproduce_ea.py`; full check:
  `python reproduce_ea.py pfsp-perturb-small pfsp-accept-small bap-op-small control`).
- `code/unit_checks.py` — identities the instruments must satisfy.
- `code/ea_reference/` — the EA 2026 per-run records used by the reproduction check.

Run from `code/`. Python 3.10+, numpy, scipy.
