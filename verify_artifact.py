"""Validate artifact integrity and numerical consistency without source posts.

Optionally verify a separately obtained upstream checkout against source hashes.
Does not certify licensing, author review, anonymity or venue compliance.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkout', type=Path)
    args = parser.parse_args()
    manifest = ROOT / 'CHECKSUMS.json'
    if manifest.exists():
        checks = json.loads(manifest.read_text())
        actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
                  and p.name != 'CHECKSUMS.json' and '__pycache__' not in p.parts}
        require(set(checks) <= actual, 'A packaged file is missing')
        extra_count = len(actual - set(checks))
        if extra_count:
            print(f'NOTE: {extra_count} additional locally created files are outside archive integrity checks.')
        for name, expected in checks.items():
            require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f'Changed file: {name}')
    s = json.loads((ROOT / 'results/disagreement/summary.json').read_text())
    v = json.loads((ROOT / 'results/disagreement/sensitivity.json').read_text())['groups']
    require(sum(s['pattern_counts'].values()) == s['eligible'] == 4562, 'Eligibility total')
    require(s['audit']['unique_matched'] + s['audit']['unmatched'] == 5303, 'Matching partition')
    require(s['audit']['unique_matched'] - s['audit']['matched_but_ineligible'] == s['eligible'], 'Eligibility partition')
    require(sum(s['released_vs_majority'].values()) == s['eligible'], 'Majority partition')
    require(v['full_test']['n'] == v['eligible']['n'] + v['excluded']['n'] == 5303, 'Coverage partition')
    require(v['released_equals_unique_majority']['n'] == s['released_vs_majority']['same'], 'Sensitivity size')
    for group, g in s['groups'].items():
        require(sum(g['released_classes'].values()) == g['n'], f'{group}: class counts')
        for name, model in g['models'].items():
            for metric in ['macro_f1', 'vote_agreement']:
                score = model['mean'][metric]
                require(0 <= score <= 1, f'{group}/{name}/{metric}: range')
                if model['ci95']:
                    lo, hi = model['ci95'][metric]
                    require(0 <= lo <= hi <= 1, f'{group}/{name}/{metric}: interval')
                runs = model['per_run']
                require(math.isclose(score, sum(r[metric] for r in runs)/len(runs), abs_tol=1e-12), 'Run mean')
            require(model['mean']['vote_agreement'] <= g['vote_ceiling'] + 1e-12, 'Vote ceiling')
        if group != 'three_way':
            for metric, contrast in s['contrasts'][group].items():
                delta = g['models']['afriberta']['mean'][metric] - g['models']['char']['mean'][metric]
                require(math.isclose(delta, contrast['mean_delta'], abs_tol=1e-12), 'Paired point contrast')
    for name, model in s['groups']['eligible_all']['models'].items():
        for metric in ['macro_f1', 'vote_agreement']:
            require(math.isclose(model['mean'][metric], v['eligible']['models'][name][metric], abs_tol=1e-12), 'Independent sensitivity agreement')
    paper = ROOT / 'paper'
    main_tex = (paper / 'main.tex').read_text()
    combined = main_tex
    for item in re.findall(r'\\input\{([^}]+)\}', main_tex):
        combined += '\n' + (paper / item).read_text()
    labels = set(re.findall(r'\\label\{([^}]+)\}', combined))
    require(set(re.findall(r'\\ref\{([^}]+)\}', combined)) <= labels, 'Undefined TeX reference')
    cited = {key for keys in re.findall(r'\\cite\{([^}]+)\}', combined) for key in keys.split(',')}
    require(cited <= set(re.findall(r'\\bibitem\{([^}]+)\}', combined)), 'Undefined bibliography citation')
    for item in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', combined):
        require((paper / item).exists(), f'Missing figure: {item}')
    table = (paper / 'tables/disagreement_models.tex').read_text()
    for m in s['groups']['eligible_all']['models'].values():
        for metric in ['macro_f1', 'vote_agreement']:
            require(f"{m['mean'][metric]:.3f}" in table, 'Model table score mismatch')
    if args.source_checkout:
        source = ROOT / 'provenance/source_manifest.json'
        if not source.exists():
            source = ROOT / 'data/source_manifest.json'
        for file in json.loads(source.read_text())['files']:
            path = args.source_checkout / file['path']
            require(hashlib.sha256(path.read_bytes()).hexdigest() == file['sha256'], f'Upstream checksum: {file["path"]}')
    print('PASS: integrity (when packaged), population accounting, aggregate scores, sensitivity, paper dependencies' +
          (', upstream source checksums.' if args.source_checkout else '.'))
    print('This is aggregate validation, not recomputation from per-item predictions.')


if __name__ == '__main__':
    main()
