"""Build an allowlisted local artifact without raw or per-item dataset records."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
FILES = [
    'prepare_public_data.py', 'baselines.py', 'train_encoder.py',
    'analyze_disagreement.py', 'review_sensitivity.py', 'build_disagreement_assets.py',
    'run_pinned_matrix.py', 'verify_artifact.py', 'build_release.py',
    'requirements-analysis.txt', 'requirements-cpu.txt', 'DISAGREEMENT_PROTOCOL.md',
    'LITERATURE_REVIEW.md', 'MANUSCRIPT_REVIEW.md', 'DATA_SHARING.md',
    'results/cpu/summary.json', 'results/disagreement/summary.json',
    'results/disagreement/sensitivity.json', 'results/disagreement/agreement_evaluation.pdf',
    'results/disagreement/agreement_evaluation.png', 'paper/main.tex',
    'paper/tables/splits.tex', 'paper/tables/disagreement_groups.tex',
    'paper/tables/disagreement_models.tex', 'paper/tables/disagreement_contrasts.tex',
    'paper/images/agreement_evaluation.pdf',
]


def main():
    dist = ROOT / 'dist'
    dist.mkdir(exist_ok=True)
    dest = dist / 'hausacs-reproducibility-candidate'
    # Refuse to traverse or replace symlink destinations.
    if dest.is_symlink():
        raise ValueError('Artifact destination must not be a symlink')
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir()
    for rel in FILES:
        source = ROOT / rel
        if source.is_symlink():
            raise ValueError(f'Refusing symlink: {rel}')
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(ROOT / 'ARTIFACT_README.md', dest / 'README.md')
    provenance = dest / 'provenance'
    provenance.mkdir()
    shutil.copyfile(ROOT / 'data/source_manifest.json', provenance / 'source_manifest.json')
    split = json.loads((ROOT / 'data/clean/split_policy.json').read_text())
    split.pop('exclusions')  # No row identifiers or linkage ledger in the package.
    (provenance / 'split_policy.json').write_text(json.dumps(split, indent=2) + '\n')
    runs = []
    for model in ['afriberta', 'mbert', 'xlmr']:
        for seed in [42, 43, 44]:
            path = ROOT / f'results/kaggle_matrix/hausacs_results/{model}_seed{seed}/summary.json'
            info = json.loads(path.read_text())
            selected = {k: info[k] for k in ['config', 'history', 'test_macro_f1', 'elapsed_seconds', 'torch', 'transformers', 'gpu']}
            selected.update(model_key=model, source_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            runs.append(selected)
    (provenance / 'model_runs.json').write_text(json.dumps({
        'environment_note': 'Recorded versions, not a complete historical GPU dependency lock; no weights included.',
        'runs': runs}, indent=2) + '\n')
    validation = json.loads((ROOT / 'results/release_validation.json').read_text())
    validation = {k: v for k, v in validation.items() if k not in ['fresh_directory', 'steps', 'archive_sha256']}
    (provenance / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    checks = {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(dest.rglob('*')) if p.is_file()}
    (dest / 'CHECKSUMS.json').write_text(json.dumps(checks, indent=2) + '\n')
    archive = dist / 'hausacs-reproducibility-candidate.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():
                item = zipfile.ZipInfo((Path(dest.name) / p.relative_to(dest)).as_posix(), date_time=(2026, 11, 7, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = 0o100644 << 16
                z.writestr(item, p.read_bytes())
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(f'{checksum}  {archive.name}\n')
    print(json.dumps({'archive': str(archive), 'sha256': checksum, 'files': len(checks) + 1,
                      'contains_dataset_rows': False, 'anonymous_submission': False,
                      'published': False}, indent=2))


if __name__ == '__main__':
    main()
