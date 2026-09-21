"""Read-only checks of the excluded preliminary values quoted in Appendix B."""
import json
from pathlib import Path
import re
import statistics as st

ROOT = Path(__file__).resolve().parent / 'historical_diagnostics'
data = json.loads((ROOT / 'results_E_anytime_seeded.json').read_text(encoding='utf-8'))
x, y = [], []
for g in data['per_grid'].values():
    assert len(g['ok_seeds']) == len(g['floor_nmse']['vals']) == len(g['c1_geo']['vals'])
    x.extend(g['floor_nmse']['vals'])
    y.extend(g['c1_geo']['vals'])
assert len(x) == 14
r = st.correlation(x, y)
assert round(r, 2) == -.71
means = [st.mean(g['floor_nmse']['vals']) for g in data['per_grid'].values()]
assert round(min(means), 2) == .22 and round(max(means), 2) == .40
assert round(data['per_grid']['32']['c1_geo']['mean'], 2) == 2.36
assert round(data['per_grid']['64']['c1_geo']['mean'], 2) == 1.37
comb = (ROOT / 'floor_check.log').read_text(encoding='utf-8')
maze = (ROOT / 'floor_E.log').read_text(encoding='utf-8')
assert 'cap=542' in comb and 'floor=0.0109' in comb
assert 'cap=306' in maze and 'floor=0.3536' in maze
print(json.dumps({'runs': len(x), 'r_floor_c1': r, 'grid_mean_floors': means,
                  'individual_floor_range': [min(x), max(x)],
                  'preliminary_caps_differ': [542, 306],
                  'not_used_in_main_results': True}, indent=2))
