"""Read-only reconstruction of the two documented FROZEN v2 source differences."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OLD = {
    'branch_E_anytime.py': '2a7d3278d80332451320727de5ecc2f40a4c432aebde4efc00780bd829a1227f',
    'branch_F_tortuosity.py': '54c8e8bbe8a8a545ff7b0243890f2addcd770af4ee8de73d25982fbb4b776586',
}
REPLACEMENTS = {
    'branch_E_anytime.py': [
        ('  * HORIZON. One GLOBAL anytime cap is set once from the maximum training-pool distance and\n'
         '    then shared unchanged by every grid. A per-grid cap would make the horizon itself',
         '  * HORIZON. One GLOBAL anytime cap shared by every grid, fixed above any plausible knee and\n'
         '    independent of any per-scene distance. A per-grid cap would make the horizon itself'),
        ('(set from the training-pool maximum; shared by all grids)',
         '(shared by all grids; distance-independent)'),
    ],
    'branch_F_tortuosity.py': [
        ("    The seed is resampled for up to `tries` batched rounds to seek at least `min_reach` of\n"
         "    the scene's free cells. A scene that still misses the threshold after the final round is\n"
         "    retained rather than silently discarded.",
         "    The seed is resampled (up to `tries` rounds, batched) until it reaches at least\n"
         "    `min_reach` of the scene's free cells, so no scene degenerates into a sealed pocket whose\n"
         "    D_geo is trivially small."),
    ],
}


class Normalize(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace(
                '(set from the training-pool maximum; shared by all grids)',
                '(shared by all grids; distance-independent)')
        return node

    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                v = node.body[0].value
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    node.body.pop(0)
        return node


def audit():
    report = {'scope': 'byte-exact historical reconstruction and normalized AST comparison',
              'files': [], 'numeric_code_unchanged': True,
              'old_manifest_was_not_modified': True}
    for name, changes in REPLACEMENTS.items():
        data = (EXP / name).read_bytes()
        current = data.decode('utf-8')
        historical = current
        for new, old in changes:
            assert historical.count(new) == 1, (name, new)
            historical = historical.replace(new, old)
        digest = hashlib.sha256(historical.encode('utf-8')).hexdigest()
        assert digest == OLD[name], (name, digest)
        trees = [ast.dump(Normalize().visit(ast.parse(s)), include_attributes=False)
                 for s in (current, historical)]
        assert trees[0] == trees[1], name
        report['files'].append({'file': name, 'current_sha256': hashlib.sha256(data).hexdigest(),
            'reconstructed_frozen_v2_sha256': digest, 'matches_frozen_v2': True,
            'normalized_ast_equal': True,
            'diff_old_to_current': ''.join(difflib.unified_diff(
                historical.splitlines(True), current.splitlines(True),
                fromfile='frozen_v2/' + name, tofile='release/' + name))})
    return report


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    result = audit()
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as f:
            f.write(text)
    print(text)
