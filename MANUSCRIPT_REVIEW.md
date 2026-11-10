# Manuscript and submission audit — 2026-11-07

Release note, 2026-11-10: the author authorized publishing the reviewed
allowlisted code, manuscript, aggregate outputs and provenance in a new
GitHub repository, separate from the older code-switching project. The
historical review below predates that publication action. Its unresolved
venue, institutional, scientific and license decisions remain unchanged;
no raw posts, individual votes or per-item predictions are authorized for
inclusion by the repository update.

## Overall assessment: share with caveats; not submission-ready

The completed analysis supports an exploratory release/evaluation audit.
No target venue has been chosen, so no venue-compliance certification, deadline,
fee, page-count approval or acceptance assessment is made. No submission,
public upload or paid compute was performed during this audit.

## Claim audit

| Claim or wording | Evidence and disposition |
|---|---|
| 5,303 test records; 4,562 eligible | Source/import hashes and complete matching/eligibility ledger; retain with denominator and selection caveats |
| 3,055 unanimous / 1,501 two-to-one / six three-way | Eligible join and summary; retain; no inference on six three-way posts |
| Overall ranking is stable | Reworded to unchanged point ordering between metrics **within eligible posts**; full-test coverage changes word/mBERT order |
| AfriBERTa outperforms character LR | Unsupported as a reliable advantage; retain only small point differences with intervals spanning zero; no equivalence claim |
| 64 released labels differ from raw majority | Retain as consistency observation, not errors; adjudication is documented in NaijaSenti but individual causes are unknown |
| Lower contested-post scores establish model failure | Unsupported causal/ambiguity conclusion; retain association, class-composition and metric-ceiling cautions |
| First disagreement-aware Hausa study / new metric | Unsupported; closest 2026 work includes Hausa AfriSenti. Explicitly disclaimed |
| Code-switching, pragmatics, calibration or individual perspectives measured | Unsupported; no gold LID/pragmatic annotation, probability outputs or individual-annotator prediction; no such claims |
| Full reproducibility from sharing archive alone | Unsupported; archive regenerates aggregates/assets. Exact recomputation needs separately obtained source and saved predictions |
| Kaggle input is currently private | Not remotely reverified; replaced with precise statement about recorded local private-input metadata |
| All hyperparameters described | Added model identifiers, weight decay, warmup, clipping and FP16; recorded model revisions included in supplementary metadata |
| No new participants means no ethics obligations | Not claimed; public-data provenance, privacy, permissions and applicable institutional/venue requirements still need author review |

The literature comparison is targeted rather than exhaustive. The contribution
remains limited in methodological novelty and should be evaluated on the
usefulness of the release-linkage and coverage audit.

## Numerical and artifact validation

`verify_artifact.py` checks population accounting, score ranges and ceilings,
run means, paired point differences, agreement with the independent sklearn
sensitivity analysis, TeX dependencies and packaged-file hashes. A separate
fresh-directory recomputation is used to check source preparation and the
saved-prediction analysis. See `results/release_validation.json` in the working
project for executed checks and outcomes; this report is not independent peer
review. No rendered manuscript pagination or venue-template visual compliance
is certified by compilation alone.

## Submission requirements and unresolved decisions

These are conditional examples from current official ACL/ARR guidance, checked
2026-11-07, not requirements imposed on every workshop or an IEEE venue.

| Item | Current state / next action |
|---|---|
| Venue and contribution track | Select an appropriate analysis/reproducibility venue; confirm its current CFP |
| Template and length | IEEEtran is the working style; ARR requires ACL style. ARR content limits are four pages for short and eight for long papers |
| Limitations placement | Present inside Discussion; ARR requires a dedicated section after the conclusion. Move when adapting to that venue |
| Review anonymity | Main TeX and artifact retain author name. ARR requires anonymized paper and supplements; prepare a review version after venue selection |
| AI assistance | Substantive framing, search, code, analysis and drafting disclosed in paper and artifact. ARR requires disclosure in acknowledgments and its checklist |
| Responsible-research form | Draft evidence is documented; author must complete the actual venue form, verify permissions and institutional requirements, and avoid unsupported approvals |
| Author/profile/service eligibility | Author list/affiliation and publication history need author confirmation; ARR requires complete OpenReview profiles and its current reviewing-policy checks |
| Submission exclusivity and prior publication | Author must confirm concurrent submissions and preprint/publication status; not inferred from local files |
| Public artifact release | Candidate is local, attributed and stripped of rows; explicit project-code license and unresolved source permissions remain decisions |
| Content verification | Author must verify citations, methods, results and AI-assisted text/code before submission |

Official sources: [ARR CFP](https://aclrollingreview.org/cfp),
[submission checklist](https://aclrollingreview.org/authorchecklist),
[Responsible NLP guidance](https://aclrollingreview.org/responsibleNLPresearch/).
The CFP is newer than parts of the authors-guideline page; use the current CFP
and the selected venue's own rules when they differ. No formatting conversion
or anonymization was applied to the user's open working draft without a target.

## Packaging boundary

The allowlisted ZIP includes analysis/training code, aggregate results, paper
sources, plots, dataset source checksums, model revisions and recorded run
settings. It excludes raw/clean data, annotation IDs/votes, source-row joins,
per-item outputs, model weights, credentials and historical reviewer material.
No third-party dataset or software license is reassigned. See DATA_SHARING.md
and the artifact README for the exact reproduction limitations.
