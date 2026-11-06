# Journal of Heuristics extension — design document (protocol)

Status: APPROVED by the author 6 Nov 2026 (decisions of Section 12 included); FROZEN from its commit to the repository: no
setting, candidate set, budget, seed, test or decision rule below changes after any new run, except through a dated
entry in Section 11 that says what changed, why, and whether any result had been seen.

Working title: "Screening adaptive parameter control before learning: null probes, decoupled probes and calibration
against trained controllers". Extension of Niebles-Atencio & Voss, "When does learned control help? An oracle
analysis of adaptive parameter control for metaheuristics on scheduling problems", Artificial Evolution (EA 2026),
LNCS, to appear.

## 1. What the conference paper established and what it left open

Established (EA 2026; numbers from `ea2026_run.log`, reproduced by `ea2026_suite.py`):
- the greedy one-step probe, read one-sidedly as an optimistic estimate of the headroom for control;
- probe gain over the best fixed value: PFSP +0.01% to +0.46% (perturbation strength and acceptance temperature,
  three scales each), BAP operator selection -0.46% to +0.66%, positive control +90.6%;
- a tabular Q controller on BAP-medium: +0.11% over the best fixed operator, the probe +0.06%.

Open, and the subject of this paper:
- Q1 How much of a probe gain is luck (best of K draws) rather than choice among values?
- Q2 Is there an instrument whose gain can only come from state dependence? (decoupled probe)
- Q3 Does a longer horizon change the verdict? (two-step instruments)
- Q4 Does the screen agree with the exact headroom where it is known? (LeadingOnes)
- Q5 Does the screen predict what trained controllers realize? (calibration)
- Q6 Where can a one-step screen be wrong? (tabu tenure, a delayed-effect parameter)
- Weaknesses of the conference protocol, fixed here: the one-sided reading holds for one step, not for a run (the EA
  controller beat the probe on BAP-medium); the tabular controller was trained and evaluated on the same instances;
  six instances per cell cannot reach significance after adjustment; no equivalence test behind "flat".

## 2. Instruments

Notation as in the EA paper: candidate set Theta = {theta_1..theta_K}; step(x, theta, omega); objective f
(minimized; LeadingOnes and NK are negated so that lower is better everywhere).

- Fixed run F(theta); best fixed theta* selected per (setting, instance) on the selection seeds (EA D5) and evaluated
  on the evaluation seeds; second baseline theta*_pool selected once per setting on a separate instance pool.
- Probe P (EA, unchanged): one trial per value per step, independent draws; the best trial is the next incumbent.
  DESCRIPTIVE (its error over a run can go either way).
- Null probe N: K trials of theta* per step; the best trial is the next incumbent. Measures the luck. DESCRIPTIVE.
- Excess gain = P vs N. NOT a measure of state dependence: with K = 2, Y* = 0 and Y' = 0.1 +/- 1 the excess gain is
  positive although theta* is best in the mean everywhere (the probe rewards variance). DESCRIPTIVE.
- Decoupled probe D_m: m trials per value per step; choose the value with the best trial mean; take ONE FRESH step
  with it (the trials only choose). One-step expectation = sum_theta Pr[choice = theta] mu_theta(x), which is at least
  mu_theta*(x) whenever theta* is mean-best at x. Implementable policy with K m + 1 steps per iteration.
  DECISION INSTRUMENT. m = 3 everywhere; m = 10 where affordable (Section 4). Ties between values are broken
  uniformly at random (with post-acceptance scores, rejected trials tie at the incumbent's cost).
- Two-step versions P2 and D2_3 on ordered pairs of values every second iteration (one decision consumes two steps);
  the one-step bound carries over only for rules that commit to both values before observing.
- Selection counts of P and D_m per value and search phase.

Verdict per cell, on D_3 (and D_10 where run) vs F(theta*): FLAT if every D_m run is equivalent within delta (TOST);
HEADROOM if any D_m is significantly better; UNDETERMINED otherwise; a D_m significantly worse is reported as such.

Propositions (proved in the paper, no runs needed): one-step bound (EA); null probe <= fixed in one step; excess gain
zero under identical distributions and <= 0 under stochastic dominance of theta*, with the counterexample above;
decoupled probe >= fixed in one step when theta* is mean-best.

## 3. Settings (nine)

| ID | Problem / metaheuristic / parameter | Theta | Budget (steps) | Tested instances | Status |
|----|-------------------------------------|-------|----------------|------------------|--------|
| S1 | PFSP, ILS, perturbation strength | {2,4,6,8,12} | 150 (20x5), 120 (50x10) | Taillard ta001-ta010 (20x5), ta041-ta050 (50x10) | new instances; EA instances for reproduction only |
| S2 | PFSP, ILS, acceptance temperature | {0,0.002,0.005,0.01,0.02} | as S1 | as S1 | as S1 |
| S3 | PFSP, tabu search on the insertion neighbourhood, tabu tenure (delayed effect) | {3,7,15,30} | 1,000 moves | as S1 | new |
| S4 | BAP (EA synthetic), LNS destroy operator, greedy reinsertion | 5 destroy operators | 300 / 250 / 150 | 20 instances per EA scale (the EA 6 + 14 new) | EA + new |
| S5 | BACASP under wind (JoS), LNS action | 54 actions; and the 6 operators | 500 | JoS reference test pool, 50 instances x 3 realizations | known outcome |
| S6 | BAP+QCAP+bay-level QCSP under wind (QCSP paper), LNS operator | 7 operators | 500 | QCSP reference test pool, 50 x 3 | known outcome |
| S7 | LeadingOnes, RLS flipping k bits (exact headroom by DP) | {1,2,4,8,16} | n^2/2 | n in {50,100}; unit = seed | new |
| S8 | NK landscapes, ILS flipping k bits + bit-flip descent | {1,2,4,8} | 2,000 | N = 100, K_NK in {0,2,4,8}, 20 landscapes each | new |
| S9 | Positive control (EA landscape), Gaussian step size | {0.5,2,8,20} | 100 | one landscape; unit = seed | EA |

Notes:
- Budgets count STEPS (incumbent updates). A probe's trials within a step do not count against the budget; they are
  its extra cost, reported separately.
- Taillard instances are generated with Taillard's published generator and seeds, checked against the published data
  of ta001 before any run. The six EA instances per scale are rerun only to reproduce the EA rows (Section 8, step 1).
- S2: the probes rank trials after the acceptance decision (EA D3), which works against high temperatures; reported.
- S5/S6: the probes run at the reference configuration; theta* is the best fixed action/operator already selected on
  the controllers' validation instances (JoS: swap, severity 1, temperature 0.005; QCSP: swap). Known outcomes:
  Section 7, PR4.
- S7 exact references: for every policy that chooses k as a function of the current LeadingOnes value, the expected
  final value after n^2/2 steps is computed by dynamic programming over (value, steps left), adapted from the
  expected-runtime analysis behind the Biedenkapp et al. (2022) benchmark; this gives the best fixed k, the optimal
  dynamic policy and the exact headroom.
- DROPPED from the earlier plan, before any run: the GA on the TEVC problem (one generation takes seconds, so the
  instruments' K-fold cost makes it infeasible at a useful sample size; and the TEVC paper is being corrected) and the
  Cordeau et al. (2005) instances (not available).

## 4. Units, seeds and replication

- Unit of every test = instance (mean over seeds and realizations); every tested cell has at least 10 instances, so
  that a paired test can reach significance after Holm (with 6 instances the smallest exact two-sided p is 0.031).
  S7 and S9 have one instance: unit = seed (100 seeds); on S7 exact values replace tests for the fixed values.
- Seeds: 5 selection + 15 evaluation seeds per (instance, value) for S1-S3; 5 + 25 for S4 and S8; S5/S6 one run per
  test instance and realization; S7 and S9 100 seeds.
- D_3 in every cell. D_10 in addition: S1-S3 at 20x5; S4 all scales; S5 on the 6 operators (not on the 54 actions:
  541 trials per step); S7; S8; S9. Not on S1-S3 at 50x10 (about 51 fixed-run equivalents per run) nor on S6 (about
  6.5 min per run); there D_3 only, and its lower power is stated with the results.
- Two-step instruments (P2, D2_3): S1 and S2 at 20x5; S4 all scales; S5 on the 6 operators (54 actions would need
  2,916 pairs per decision) and S6, both on 20 instances x 1 realization; S7; S8 with K_NK in {0,4}; S9.

## 5. Trained controllers (realized gain)

Each trained, where needed, on an instance pool DISJOINT from the evaluation instances:
- C1 adaptive weights (segment 50, reaction 0.5, scores 5/3/1); no training;
- C2 UCB1 over Theta; no training;
- C3 tabular Q (state: stagnation {0, 1-9, >= 10} x last step improved; 6 states), 400 episodes;
- C4 DQN: the existing JoS and QCSP controllers on S5 and S6; new DQNs on S4-medium and S7 (n = 50) with the
  search-dynamics state of the JoS controller, 3 seeds x 4,000 episodes.
Realized gain = paired comparison with F(theta*) and with F(theta*_pool). Training time is recorded.

## 6. Statistics and decision rules

- Wilcoxon signed-rank paired on the unit of the cell, for P, N, D_m and each controller vs F(theta*), and P vs N.
- TOST at delta = 0.5% of the best-fixed mean (S1-S6, S8, S9); delta = 1 LeadingOnes value (n = 50) or 2 (n = 100)
  for S7.
- Verdict as in Section 2. Holm adjustment within each table.
- Calibration: Spearman correlation between the D_m gain (D_10 where run, else D_3) and the best realized controller
  gain across cells; count of cells with a FLAT verdict in which some controller is significantly better than
  F(theta*) (the screen's false "no"), each reported.

## 7. Written predictions (before any new run)

- PR1 On S1, S2, S4: in every cell where the probe gain is positive, the null-probe gain is at least half of it.
- PR2 On S1, S2, S4 the verdict is flat or undetermined in every cell, and the two-step instruments change no
  verdict. (The clause on S9 was withdrawn on 6 Nov, before freezing: S9 was used in the pilot, Section 12.)
- PR3 On S7 the exact headroom exceeds 2 delta and the verdict is headroom. A flat verdict on S7 falsifies the screen.
- PR4 (known outcomes, declared as such) On S5 and S6 the verdict is flat or undetermined. JoS: the DQN is worse than
  the best fixed action at 3 of 14 points and better at none (Holm); QCSP: worse than the best fixed operator at
  14 of 21 points and better at none (Holm). These are not blind tests. A headroom verdict would be a false "yes"
  relative to the trained controllers, and is reported as such.
- PR5 On S3 the one-step verdict is flat; whether it is a false "no" depends on whether a reactive tenure rule beats
  the best fixed tenure (no prediction on that).
- PR6 On S8 the D_m gain grows with K_NK; flat at K_NK = 0.
- PR7 Spearman correlation > 0; no flat cell with a controller significantly better than F(theta*), except possibly
  in S3.

## 8. Order of work

1. Refactor `ea2026_suite.py` into a package with one step interface per setting; reproduce `ea2026_run.log` rows
   for S9 and for the small scales of the EA PFSP and BAP settings exactly (same seeds) before any new code is used.
2. Add N, D_m, P2 and D2_3, the controllers C1-C3, the Taillard generator (checked on ta001), S3, S7 (with DP), S8,
   and the 14 new instances per scale for S4.
3. Unit checks: DP vs simulation on LeadingOnes (n = 20); N and D_m with K = 1 equal the fixed run; P2 with K = 1
   equals two fixed steps; ta001 data match.
4. Pilots (timing only, recorded in Section 11); then full runs. A budget is changed only if a cell is infeasible,
   and only through Section 11 before its full run.
5. S5/S6: instrument drivers on the JoS and QCSP code (extensions of `8_fixedop.py` and `10_fixedop.py`).

## 9. Compute (estimate from the EA timings; the pilot fixes it before the full runs)

Per instance and seed the instruments cost about 3K fixed-run equivalents for F, P and N, plus 3K + 1 for D_3 and
10K + 1 for D_10 where run. From the EA run times (about 1 s per run at 20x5 and 9 s at 50x10 per mechanism, about
1 s for BAP-large): S1-S2 about 40 core-hours (30 of them at 50x10), S3 about 15, S4 about 12, S5 and S6 about 12
together, S7-S9 about 12, two-step instruments (P2 and D2_3) about 20, controllers about 15. Total about 125-145 core-hours,
within the 120-160 announced on 3 Nov (the D_10 and D2_3 runs decided on 6 Nov account for the increase over the
80-110 of 4 Nov). This is the estimate the full runs are planned against; the pilot timings only check it.

## 10. Reporting

Every number in the paper is produced by a script from the CSV records; tables are generated, not typed. Known-outcome
settings (S5, S6) are labelled as such wherever they appear. Negative and null results are reported in full.

## 11. Changes after freezing

(none yet)

## 12. Pre-freeze pilot, 5 Nov 2026 (protocol NOT yet frozen; recorded so that its influence on the design is visible)

Code: `code/screen/instruments.py` (F, P, N, D_m, P2, D2_m on one step interface), `code/screen/ea_core.py`
(byte-identical copy of the EA suite), `code/reproduce_ea.py`, `code/unit_checks.py`.

- Reproduction: F and P reproduce the EA records EXACTLY (0 mismatches): control 4,350 runs, bap-op-small 1,050 runs;
  pfsp-perturb-small and pfsp-accept-small, instance 0: 175 runs each, 0 mismatches (`code/repro_pfsp.log`).
- Unit checks: N(K=1) equals P({theta}) exactly; D_1({theta}), N(K=1), P2({theta}), D2_1({theta}) have the law of
  F(theta) on the control (Mann-Whitney p = 0.27-0.59, 400 seeds).
- Tie rule: with post-acceptance scores most trials of the control are rejected and tie at the incumbent's cost; the
  first implementation of D_m broke ties towards the first value and was trapped at the smallest step (mean 0.468 vs
  F(8) 0.091). Ties are now broken uniformly at random (D_m and D2_m only; P keeps the EA rule for reproduction).
- Positive control (S9), 200-400 seeds: F(8) 0.081-0.091; P 0.011 (probe gain 87%); N(K=4, theta* = 8) 0.012
  (null gain 85%): about 98% of the probe gain on the positive control is best-of-K luck; P vs N still differs
  (Mann-Whitney p = 1.8e-24). D_1 0.124, D_3 0.104 (both slightly WORSE than F(8), p about 0.04), D_10 0.075 (better,
  p = 7e-5).
- Consequences for the design (DECIDED by the author on 6 Nov 2026, before freezing; Option A):
  (a) D_1 and D_3 lack power where the information in one trial is small (most trials rejected): m = 10 is needed on
      S9. Decided: D_3 everywhere, D_10 where affordable (Section 4); D_1 dropped.
  (b) S9 can no longer serve as a blind test: its outcome under the instruments is now known. PR2's clause on S9 is
      withdrawn; S9 stays as an illustration, and the ground truth for the screen is S7 (LeadingOnes, exact).
  (c) The EA conference result on the positive control (+90.6% probe gain) is almost entirely luck under the null
      probe. The JoH paper reports this as a correction of the conference reading.
