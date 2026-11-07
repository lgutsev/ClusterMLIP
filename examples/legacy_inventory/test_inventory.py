"""Focused validation for non-destructive three-warehouse inventory."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from cluster_mlip.io import write_extxyz, write_manifest
from cluster_mlip.models import Atom, Record


def main():
    script = Path(__file__).with_name('inventory.py')
    subprocess.run([sys.executable, str(script), '--self-test'], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, campaigns, output = root/'source', root/'campaigns', root/'audit'
        source.mkdir()
        text = '# UBPW91/6-311+G*\n\nfixture\n\n0 3\nFe 0 0 0\nFe 2 0 0\n\n'
        (source/'fe2.com').write_text(text)
        (source/'same.com').write_text(text)
        (source/'unsupported.bin').write_bytes(b'unknown')
        record = Record('old', 'old.com', [Atom('Fe',0,0,0), Atom('Fe',2,0,0)], 0,3,'unknown',route='# UBPW91/6-311+G*')
        for name in ('FenOm_Warehouse','FenOm_Warehouse2','General_Warehouse'):
            folder = campaigns/name/'extracted'
            folder.mkdir(parents=True)
            write_extxyz([record], folder/'seeds.extxyz')
            write_manifest([record], folder/'manifest.csv')
        before = {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
        cmd = [sys.executable,str(script),'--source',str(source),'--campaigns',str(campaigns),'--output',str(output)]
        subprocess.run(cmd, check=True)
        summary = json.loads((output/'summary.json').read_text())
        assert summary['records'] == 2  # do not drop aliases through extract's de-duplication
        assert len(summary['reference_inputs']) == 3
        assert summary['overlap_pairs'] == 7  # 2 x 3 historical + one incoming pair
        assert summary['file_status']['unsupported'] == 1
        assert all(Path(p).read_bytes() == data for p,data in before.items())
        assert subprocess.run(cmd, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode != 0
        (campaigns/'General_Warehouse'/'extracted'/'manifest.csv').unlink()
        cmd[-1] = str(root/'missing_reference_audit')
        assert subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode != 0
        assert not (root/'missing_reference_audit').exists()
    print('PASS: full extraction, all three overlaps, alias retention, unchanged sources, nonoverwrite, missing-input guard')


if __name__ == '__main__':
    main()
