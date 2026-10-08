import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_self_test():
    result = subprocess.run([sys.executable, str(HERE / "root_check.py"), "--self-test"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "self-test passed" in result.stdout


def test_generate_package(tmp_path):
    subprocess.run([sys.executable, str(HERE / "generate_package.py"), str(tmp_path)], check=True,
                   capture_output=True)
    pkg = tmp_path / "35_clustermlip_electronic_root_check"
    for name in ("root_check.py", "run_root_check.slurm", "README.md", "export.list", "package.json"):
        assert (pkg / name).is_file()
    assert b"\r\n" not in (pkg / "run_root_check.slurm").read_bytes()
