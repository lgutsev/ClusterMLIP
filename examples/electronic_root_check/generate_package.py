#!/usr/bin/env python3
"""Generate the electronic-root check package (proposed 35) from the owning ClusterMLIP repository."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

NAME = '35_clustermlip_electronic_root_check'


def write_lf(path: Path, text: str) -> None:
    path.write_bytes(text.replace('\r\n', '\n').encode())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path, help=f'parent directory; the package is created as {NAME}/')
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    dest = args.destination / NAME
    dest.mkdir(parents=True, exist_ok=False)
    for name in ('root_check.py', 'run_root_check.slurm', 'README.md'):
        write_lf(dest / name, (here / name).read_text(encoding='utf-8'))
    (dest / 'logs').mkdir()
    write_lf(dest / 'logs' / 'README.txt', 'Slurm logs land here.\n')
    write_lf(dest / '.gitignore', 'outputs/\nlogs/*\n!logs/README.txt\n')
    write_lf(dest / 'export.list', 'outputs/*\n')
    hashes = {str(f.relative_to(dest)).replace('\\', '/'): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(dest.rglob('*')) if f.is_file()}
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = ''
    write_lf(dest / 'package.json', json.dumps({
        'package': NAME, 'owner': 'ClusterMLIP',
        'generator': 'examples/electronic_root_check/generate_package.py',
        'depends_on': '26_clustermlip_legacy_inventory (run_1075007)',
        'files_sha256': hashes, 'submitted': False, 'source_commit': commit}, indent=2) + '\n')
    print(dest)


if __name__ == '__main__':
    main()
