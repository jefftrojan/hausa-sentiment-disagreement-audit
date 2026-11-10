# Literature and methodological review — 2026-11-07

## Decision

Keep the study as an exploratory, reproducible evaluation audit. Broad claims
about introducing disagreement-aware evaluation or first applying it to Hausa
are not defensible. The closest retrieved work already includes Hausa
AfriSenti. A short empirical or reproducibility paper is a more plausible
format than a new-method paper; acceptance and venue fit are not established.

## Primary-source comparison

| Work | Relevant overlap | Boundary for this paper |
|---|---|---|
| [NaijaSenti, 2022](https://aclanthology.org/2022.lrec-1.63.pdf), §4.2–4.3 | Documents individual annotations, adjudication and future disagreement-based learning | Released labels need not be raw majorities; this audit cannot diagnose errors from differences alone |
| [AfriSenti, 2023](https://aclanthology.org/2023.emnlp-main.862/) | Existing source benchmark, multilingual models and annotation process | We audit a pinned release with modified training/dev splits, rather than introduce a corpus |
| [Learning with Disagreements, 2021](https://aclanthology.org/2021.semeval-1.41/) | Established multi-annotation evaluation | No claim of inventing the evaluation paradigm |
| [Fornaciari et al., 2021](https://aclanthology.org/2021.naacl-main.204/) | Uses soft annotation distributions in training | Our fixed hard predictions do not test soft-label learning |
| [LeWiDi, 2023](https://aclanthology.org/2023.semeval-1.314/) | Shared training/evaluation framework with soft evaluation | Our empirical hard-decision vote agreement is narrower |
| [Rizzi et al., 2024](https://aclanthology.org/2024.nlperspectives-1.9/) | Studies disagreement-aware metric properties | No new metric or proper probabilistic scoring claim |
| [LeWiDi, 2025](https://aclanthology.org/2025.nlperspectives-1.16/) | Soft-label and individual-perspective evaluation | We predict neither full distributions nor individual annotator perspectives |
| [Belay et al., 2026 preprint](https://arxiv.org/html/2605.09955v1), §4.1, Table 1, §7 | Agreement-based clustering includes Hausa AfriSenti and models annotator perspectives | Closest overlap; blocks a first-Hausa-disagreement claim. Our distinction is release linkage, coverage selection, fixed predictions and paired ranking checks, not a competitive new method |

## Search scope and limits

Searched combinations of Hausa, AfriSenti, disagreement, individual labels,
soft labels, soft-label evaluation and Learning with Disagreements. Followed
primary ACL and arXiv sources, including 2025/2026 results. This is a targeted
search, not a systematic review, exhaustive priority certificate or verification
of every study cited by the retrieved papers. The 2026 clustering source was
checked beyond its abstract for explicit Hausa inclusion. Its different splits
and training targets prevent direct numerical score comparison.

## Methodological review

Assessment: share with caveats as an exploratory analysis. Do not describe this
as independent peer review or submission readiness.

Verified: the 4,562 eligible and 741 excluded IDs partition all 5,303 test IDs;
saved CPU and nine encoder predictions preserve test ordering and labels;
secondary metrics independently reproduce primary results using scikit-learn
F1; fixed references are retained across sensitivity checks. The missing
split-table inclusion in the TeX was repaired.

The added `review_sensitivity.py` uses sklearn F1 and direct vote counting,
without the primary analysis' metric function. Its results are in
`results/disagreement/sensitivity.json`, with source/prediction hashes.
These checks were selected after reviewing the primary findings. They are
descriptive, have no added confidence intervals and do not replace the primary
analysis. No training, label correction or test-based model selection occurred.

Coverage matters: word LR exceeds mBERT on the full test set (0.7320 vs 0.7289)
but falls below it on eligible posts (0.7263 vs 0.7296). Thus the unchanged
ranking between metrics is conditional on a fixed eligible population. On
the excluded complement, scores are 0.7473 and 0.6902 respectively. Do not
interpret the aggregate eligible sample as all released test posts.

On 4,492 eligible posts where released labels equal unique raw majorities,
the eligible model ordering is preserved under both metrics. AfriBERTa's
vote agreement is 0.7270 versus character LR's 0.7225. Excluding the differing
labels and ties does not remove the small point advantage; significance is
not established by this check.

Remaining caveats: only three encoder seeds, fixed classical fits, dependence
from repeated annotators, no calibrated probabilities, selective vote coverage,
one benchmark, exploratory comparisons and limited methodological novelty.
NaijaSenti's documented adjudication explains why majority disagreement can
be legitimate; it does not establish the cause of each observed discrepancy.

## Supported positioning

On a pinned Hausa release, evaluation coverage can alter point rankings even
when changing from released-label macro-F1 to vote agreement leaves rankings
stable within the eligible population. A reproducible account must distinguish
population selection, label aggregation, metric ceilings and uncertain model
differences. This is the contribution to defend, without claiming a novel
learning method, causal explanation or reliable encoder superiority.
