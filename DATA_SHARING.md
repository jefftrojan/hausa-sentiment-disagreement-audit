# Artifact sharing decisions — checked 2026-11-07

This is a conservative packaging decision, not a finding that any source has
prohibited redistribution or that all permissions have been legally resolved.

## Source notices

The pinned [AfriSenti GitHub repository](https://github.com/afrisenti-semeval/afrisent-semeval-2023)
README states CC BY 4.0. Its exact imported README checksum is recorded in
`provenance/source_manifest.json` in the package (`data/source_manifest.json`
in the working project). The related [NaijaSenti Hugging Face card](https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter)
currently states CC BY-NC-SA 4.0. These are different distributions; neither
notice is silently treated as superseding the other. We have no documented
maintainer clarification, institutional review determination, or new consent.
Do not infer any of these from public availability or lack of recruitment.

## Included in the local sharing candidate

| Material | Treatment |
|---|---|
| Project analysis, training and packaging code | Included; copyright/license grant for this project's original code remains for the author to select |
| Aggregate scores, intervals, class/group counts, plots, tables | Included as research outputs, with source attribution and scope limitations |
| Source URLs, pinned revision, SHA-256 file hashes | Included; no raw text, source tweet IDs or text hashes exported |
| Model identifiers/revisions, hyperparameters, aggregate run metrics, recorded software versions | Included; no model weights or external pretrained assets |
| Manuscript and protocol | Included; manuscript retains author name and is not an anonymized submission |

Aggregate publication should still be reviewed under the selected venue and
institution's requirements. The archive was initially prepared locally.
On 2026-11-10 the author authorized a new GitHub repository containing only the
allowlisted code, manuscript, aggregate results and provenance. This does not
authorize the omitted per-item records, select a new project-wide license,
or submit a paper. The earlier code-switching repository remains separate.

## Omitted pending release-specific permission review

Raw/clean posts, individual votes, annotator identifiers, source-row joins,
per-item predictions and labels, per-item exclusion ledgers, reviewer packs,
unrelated datasets, credentials, Kaggle account metadata and model weights
are omitted. Derived per-item records remain linkable to the source even if
post text is removed; absence of raw text alone is not anonymization.

Readers obtain the exact data directly from the original release and verify
checksums. Recomputing the exact reported scores also needs the saved CPU and
nine encoder prediction files, whose hashes are provided. Those files are
retained locally and not included in the candidate. Authorized retraining is
an alternative, with expected hardware/software nondeterminism; it is not
guaranteed to reproduce the exact saved predictions or confidence intervals.

## Attribution and dependencies

Cite NaijaSenti (Muhammad et al., LREC 2022) and AfriSenti (Muhammad et al.,
EMNLP 2023), plus the models referenced in run metadata. Record preprocessing
as a modification, not a new corpus. Third-party Python packages and TeX
classes are dependencies, not relicensed or vendored here. Source dataset
notices do not provide a license for newly authored project code. The archive
is useful for review now; an explicit project-code license is a remaining
public-release decision.
