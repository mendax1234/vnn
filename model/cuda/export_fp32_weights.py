"""Export the original V2 checkpoint with BN folding and no Q6.10 rounding.

Requires torch and numpy. Uses the existing V2 exporter/model definition.
"""
import argparse
import importlib.util
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=here.parent / 'checkpoints/cnn.pth')
    parser.add_argument('--out-dir', type=Path, default=here)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('v2_export', here.parent / 'export_weights.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    exporter.export(str(args.checkpoint.resolve()), str(args.out_dir.resolve()), 'float')


if __name__ == '__main__':
    main()
