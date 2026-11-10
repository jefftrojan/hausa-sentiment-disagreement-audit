# Disagreement-aware evaluation pivot — 2026-11-07

This is an exploratory secondary analysis of existing data and fixed model
predictions. No new participants, labels, model training or test-tuned settings.
The baseline test results were examined before this pivot. This document is
written before computing the disagreement-stratified model results; it is not
preregistration and does not establish novelty.

## Questions

1. How do model comparisons depend on unanimous versus contested annotations?
2. Does ranking by agreement with individual votes differ from ranking by
   macro-F1 against the released label?
3. How often does the released label differ from the unique raw-vote majority?
   Such differences are not automatically errors: adjudication may explain them.

## Matching and eligibility

Join source annotation text to the frozen 5,303-post test split by Unicode
NFKC + case-folding + whitespace collapse. No fuzzy matching or mention/URL
removal. Require exactly one source row for each included post. Preliminary
structural counts: 5,075 uniquely matched; 228 unmatched. Use only posts with
three nonmissing labels in {negative, neutral, positive}, supplied by three
distinct nonmissing source annotator IDs. This yields 4,562 eligible posts.
Archive all exclusions; never map MIXED/INDETERMINATE to neutral or impute votes.
Report excluded posts and their vote labels separately as a coverage limitation.

Groups: unanimous (3 same; preliminary n=3,055), two-to-one (n=1,501),
three-way (n=6). Three-way results are descriptive and too sparse for subgroup
claims. This is observed agreement among three available annotators, not a
population-level distribution or a gold measure of inherent ambiguity.

## Measurements

- Macro-F1/accuracy against the released three-class sentiment label.
- Vote agreement V = mean_i q_i(prediction_i), where q_i is the empirical
  distribution of the three eligible votes. This evaluates hard decisions;
  it is not a model probability, calibration score or distributional prediction.
- Available-label ceiling C = mean_i max_c q_i(c). Unanimous posts have ceiling
  1; two-to-one posts 2/3; three-way posts 1/3. Cross-group differences in V
  partly follow this ceiling by construction and cannot alone show model failure.
- Mean V - 1/3 above a uniform random decision, and C - V ceiling shortfall.
- Released-label versus unique-majority consistency; ties remain ties.

Compare majority/word/character baselines and all nine existing encoder runs.
Report encoder means over the three individual runs, not ensemble predictions.
Primary follow-up contrast: AfriBERTa minus fixed character baseline on V and
macro-F1, overall and within groups. This choice follows earlier baseline
inspection and is exploratory. Report other model rankings without selecting
only a favorable metric, group or seed. No equivalence or causal claims.

## Uncertainty and scope

Use 2,000 paired post-bootstrap replicates, seed 42, stratified jointly by
released class and agreement pattern. Resample encoder seeds (42,43,44)
jointly across encoders; CPU predictions remain fixed. A post and its three
votes stay together. Do not treat three votes or three model seeds as new
independent test posts. Preserve all point results and unadjusted intervals;
three training seeds limit precision. No new hyperparameter selection.

Do not infer annotator demographics, motives or errors from IDs. Sparse or
unavailable annotator votes are not silently discarded from the data audit.
Do not claim confidence calibration, soft-label cross-entropy, or uncertainty
recognition without probabilistic outputs. Disagreement-aware evaluation is
established prior work; the candidate contribution is a reproducible Hausa
secondary analysis, subject to a full novelty comparison.

The annotation-recruitment workflow is no longer required for this pivot.
Existing reviewer files are retained as prior project artifacts, not presented
as collected responses or as a dependency for this paper.

## Post-analysis methodological review addendum — 2026-11-07

After primary subgroup results, add descriptive checks on the full test set,
the 741-post excluded complement, and 4,492 eligible posts whose released
label equals the unique raw majority. Keep original labels and predictions;
do not refit, replace labels or attach new significance claims. Record these
separately in `results/disagreement/sensitivity.json` using
`review_sensitivity.py`. This addendum is retrospective to the primary
analysis and must not be described as a pre-analysis specification.
