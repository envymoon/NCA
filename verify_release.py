"""Read-only manifest verification and saved-number recomputation."""
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
manifest = ROOT / 'RELEASE_MANIFEST.sha256'
count = 0
for line in manifest.read_text(encoding='utf-8').splitlines():
    expected, name = line.split(maxsplit=1)
    path = ROOT / name
    assert ROOT in path.resolve().parents, name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    count += 1
print(f'Manifest: {count}/{count} passed', flush=True)
subprocess.run([sys.executable, '-B', str(ROOT / 'experiments/camera_ready/check_manuscript_numbers_636.py')], check=True)
print('Saved-number checks passed. No training or GPU replay was performed.')
