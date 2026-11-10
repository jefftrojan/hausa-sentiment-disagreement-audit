"""Print or execute the recorded nine-run matrix; never accesses Kaggle credentials.

Requires an independently provisioned compatible CUDA environment. The recorded
Torch/Transformers versions are metadata, not a complete training-image lock.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/clean')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, default=ROOT / 'provenance/model_runs.json')
    parser.add_argument('--execute', action='store_true', help='Actually train all nine GPU runs; default is a plan only')
    args = parser.parse_args()
    runs = json.loads(args.metadata.read_text())['runs']
    assert len(runs) == 9
    for run in runs:
        config = run['config']
        command = [sys.executable, str(ROOT / 'train_encoder.py'), '--data', str(args.data),
                   '--output', str(args.output), '--model', run['model_key'],
                   '--seed', str(config['seed']), '--epochs', str(config['epochs']),
                   '--revision', config['revision']]
        print(json.dumps({'execute': args.execute, 'command': command}), flush=True)
        if args.execute:
            # The trainer records actual input hashes and refuses prior run folders.
            import hashlib
            for split, expected in config['input_sha256'].items():
                actual = hashlib.sha256((args.data / f'{split}.jsonl').read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f'{split}: prepared data differs from recorded run')
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
