#!/usr/bin/env python3
"""Generate package 26 from the owning ClusterMLIP repository."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    args.destination.mkdir(parents=True, exist_ok=False)
    for name in ('inventory.py', 'run_inventory.slurm', 'README.md'):
        shutil.copyfile(here / name, args.destination / name)
    runtime = args.destination / 'runtime' / 'cluster_mlip'
    runtime.mkdir(parents=True)
    (runtime / '__init__.py').write_text('')
    for name in ('models.py', 'io.py', 'gaussian.py'):
        shutil.copyfile(repo / 'src' / 'cluster_mlip' / name, runtime / name)
    (args.destination / 'logs').mkdir()
    (args.destination / 'logs' / 'README.txt').write_text('Slurm logs land here.\n')
    (args.destination / '.gitignore').write_text('outputs/\nlogs/*\n!logs/README.txt\n')
    (args.destination / 'export.list').write_text('outputs/*\n')
    hashes = {str(f.relative_to(args.destination)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in args.destination.rglob('*') if f.is_file()}
    (args.destination / 'package.json').write_text(json.dumps({
        'package': '26_clustermlip_legacy_inventory', 'owner': 'ClusterMLIP',
        'generator': 'examples/legacy_inventory/generate_package.py',
        'source_root': '/ddnB/project/ramu/gutsev/glg/', 'files_sha256': hashes,
        'submitted': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
