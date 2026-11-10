# Hausa sentiment evaluation: reproducibility artifact

Aggregate artifact prepared 2026-11-07 and reviewed for the authorized new
GitHub repository on 2026-11-10. No source posts, annotation rows, model
weights, credentials or per-item predictions are included. See DATA_SHARING.md.
This archive is not an anonymized submission or a claim of license clearance.

## What can be reproduced immediately

Use Python 3.9 or newer in an isolated environment. Tested analysis environment:
CPython 3.9.6 with the versions in requirements-analysis.txt.

```bash
python verify_artifact.py
python -m pip install -r requirements-analysis.txt
python build_disagreement_assets.py
```

The first command checks archive hashes, populations, numerical consistency,
sensitivity results and TeX dependencies using the standard library. The last
command regenerates three tables and the figure from aggregate summaries;
it does not recompute model metrics. Font/PDF metadata differences can change
rendered-file hashes across systems. Run integrity verification before rebuilding.
The manuscript uses IEEEtran and ordinary LaTeX packages, supplied by the
reader's TeX environment. No separate compiled manuscript PDF is included.

## Data acquisition and preparation

Obtain the original repository separately; inspect its terms before use:

```bash
git clone https://github.com/afrisenti-semeval/afrisent-semeval-2023 afrisenti-source
git -C afrisenti-source checkout 5aec3cfcf87b59f885e98641f72e52182776429d
python verify_artifact.py --source-checkout afrisenti-source
python prepare_public_data.py --source afrisenti-source
python -c 'from baselines import clean_splits; clean_splits()'
```

The importer also creates historical pilot artifacts locally; no new labels
are collected and those artifacts are not needed for this study. Preparation
should yield train 13,957, dev 2,664, test 5,303, with 228 training/dev exclusions.
Compare prepared file SHA-256 values with provenance/model_runs.json and the
aggregate CPU summary before evaluating. Source paths and checksums are in
provenance/source_manifest.json; the duplicate policy is in
provenance/split_policy.json with row-level ledgers omitted.

## Exact saved-prediction analysis

This mode requires separately authorized access to the saved predictions:

- results/cpu/test_predictions.csv
- results/kaggle_matrix/hausacs_results/{afriberta,mbert,xlmr}_seed{42,43,44}/test_predictions.jsonl

Their expected checksums are in results/disagreement/summary.json under
prediction_sha256. Restore them to these paths, after preparing the source data.

```bash
python analyze_disagreement.py
python review_sensitivity.py
python build_disagreement_assets.py
```

This reconstructs the 4,562-post join, exclusion ledger, scores and 2,000 paired
bootstrap replicates; review_sensitivity.py independently uses sklearn F1.
The locally generated data/ joins and item_scores.csv contain derived item-level
records and are not automatically cleared for redistribution. Preserve the
packaged results before running commands that replace local outputs.

## Retraining alternative

```bash
python baselines.py
python run_pinned_matrix.py --output results/kaggle_matrix/hausacs_results
# Only in an independently provisioned CUDA environment, to actually train:
python run_pinned_matrix.py --output results/kaggle_matrix/hausacs_results --execute
```

The default matrix command prints the plan and spends no GPU compute. Training
requires compatible torch, transformers, sentencepiece, huggingface_hub and
scikit-learn. Original runs recorded torch 2.11.0+cu130, transformers 4.57.1,
Tesla T4 and sklearn 1.6.1; a complete historical GPU environment lock was not
captured. The model revisions, seeds, input hashes and measured runtimes are
in provenance/model_runs.json. CPU selected C and candidate scores are in
results/cpu/summary.json. GPU and CPU retraining can vary across environments;
training scripts refuse existing nonempty encoder output folders. No model
weights are bundled. No new GPU execution was performed to package this artifact.

## Evidence and limits

See DISAGREEMENT_PROTOCOL.md for original and retrospective checks,
LITERATURE_REVIEW.md for prior-work comparison, and MANUSCRIPT_REVIEW.md for
the claim audit and outstanding submission decisions. No reliable encoder
superiority, equivalence, causal effect, verified code-switching/pragmatics,
or first-Hausa-disagreement claim is made. AI assistance extended to research
framing, search, code, analysis and drafting; the author must verify and accept
responsibility for the final work. No AI system is listed as an author.

Rebuild the artifact from a populated working project with build_release.py.
The candidate itself omits private inputs used by that packaging command.

In the new GitHub repository these instructions are REPRODUCING.md; README.md
provides the study overview. No project-wide code license is newly selected
by the repository update. Dataset notices and original-code rights remain separate.
