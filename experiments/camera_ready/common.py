"""Shared reporting helpers for the post-review experiments."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
sys.path.insert(0, str(EXP))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {'n': 0, 'mean': None, 'std': None, 'cv': None}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return {'n': len(values), 'mean': mean, 'std': std,
            'cv': std / mean if std is not None and mean else None}


def verify_manifest(path):
    path = Path(path)
    failures = []
    count = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        target = path.parent / name
        count += 1
        actual = digest(target) if target.exists() else None
        if actual != expected.lower():
            failures.append({'file': name, 'expected': expected, 'actual': actual})
    return {'path': str(path), 'count': count, 'failures': failures}


def provenance():
    paths = [HERE / 'protocol.md', EXP / 'branch_F_tortuosity.py',
             EXP / 'branch_E_anytime.py', EXP / 'policy_discrete.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}
