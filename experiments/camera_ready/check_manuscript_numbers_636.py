"""Read-only recomputation of selected camera-ready claims from saved rows."""
import json
import math
import statistics as st
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent

def read(name):
    return json.loads((EXP / name).read_text(encoding="utf-8"))

def close(a, b, tol=1e-10):
    assert abs(a - b) <= tol, (a, b)

lat = read("camera_ready/latency_full_population/results.json")
for mode in ("static", "dynamic"):
    rows = [r for g in lat["per_grid"].values() for r in g[mode]["rows"]]
    flat = st.mean(st.mean(r["flat_us"]) for r in rows)
    geo = st.mean(st.mean(r["geo_us"]) for r in rows)
    ref = lat["pooled"][mode]
    assert len(rows) == 762
    close(flat, ref["flat_us"]["mean"])
    close(geo, ref["geo_us"]["mean"])
    close(1 - geo / flat, ref["saving_fraction"])
    print("latency", mode, len(rows), flat, geo, 100 * (1 - geo / flat))

g48 = sum((read(f"camera_ready/{folder}/results.json")["per_seed"]
           for folder in ("g48_v1", "reference_g48_v1")), [])
cs = [r["aggregate_c1"] for r in g48 if r["aggregate_c1"] is not None]
assert len(g48) == 5 and len(cs) == 3
print("G48 conditional", len(cs), st.mean(cs), st.stdev(cs))

width = read("camera_ready/analysis_full_population.json")["width"]["per_seed"]
for r in width:
    p = r["heldout_policy_tail"]
    assert p["f"] == .95
    close(1 - p["geo_cost"] / p["flat_cost"], p["saving_fraction"])
    close(100 * (p["geo_coverage"] - p["flat_coverage"]), p["coverage_difference_pp"])
    print("width", r["seed"], p["f"], 100*p["saving_fraction"], p["coverage_difference_pp"])

for game, g in read("eval_vglc.json")["games"].items():
    for ckpt, c in g["ckpts"].items():
        rows = c["rows"]
        assert len(rows) == g["n_eligible"]
        conv = [r for r in rows if r["knee"] is not None]
        close(len(conv)/len(rows), c["coverage_ceiling"])
        for k in ("dgeo", "dfree", "deuc"):
            vals = [r["knee"]/max(r[k], 1) for r in conv]
            close(st.stdev(vals)/st.mean(vals), c["cv"][k])
        p = c["calibration"]["per_grid"]
        bs = [max(1, math.ceil(p["geo_c"] * max(r["dgeo"], 1))) for r in rows]
        gc = sum(r["knee"] is not None and r["knee"] <= b for r,b in zip(rows,bs))/len(rows)
        fc = sum(r["knee"] is not None and r["knee"] <= p["flat_T"] for r in rows)/len(rows)
        close(st.mean(bs),p["geo_cost"])
        close(gc,p["geo_cov"])
        close(fc,p["flat_cov"])
        print("VGLC",game,ckpt,len(rows),gc,fc,st.mean(bs))
aud = read("audit_full_population.json")
eligible = [r for r in aud["rows"] if not r["clamped"]]
assert len(eligible) == 3810
for g in (16, 24, 32):
    for knee in ("knee_tail", "knee_mean"):
        report = {}
        for ruler in ("dgeo", "dfree", "deuc"):
            cvs = []
            for seed in range(5):
                ratios = [r[knee] / max(r[ruler], 1) for r in eligible
                          if r["grid"] == g and r["seed"] == seed and r[knee] is not None]
                cvs.append(st.stdev(ratios) / st.mean(ratios))
            report[ruler] = (round(100*st.mean(cvs)), round(100*st.stdev(cvs)))
        print("Table II eligible CV", g, knee, report)

levels = read("results_L_clean_G32.json")["levels"]
for level in levels:
    geometry = {r["scene_id"]: r for r in level["scene_geometry"]}
    for run in level["runs"]:
        population = [r for r in run["scene_knees"] if r["eligible"]]
        attained = [r for r in population if r["tail_knee"] is not None]
        assert len(population) == run["eligible_scenes"]
        close(len(attained)/len(population), run["tail_unconditional_coverage"])
        for short, field in (("geo", "Dgeo"), ("free", "Dfree"), ("euc", "Deuc")):
            ratios = [r["tail_knee"]/max(geometry[r["scene_id"]][field], 1) for r in attained]
            close(st.mean(ratios), run["tail_ratio"][short]["mean"])
            # The frozen L implementation uses population SD within scenes.
            close(st.pstdev(ratios)/st.mean(ratios), run["tail_ratio_cv_conditional"][short])
        print("Table VI", level["het"], run["model_seed"], len(attained), len(population))

# Paired sensitivity is distinct from the within-level populations in Table VI.
control, transfer = levels
for cr, tr in zip(control["runs"], transfer["runs"]):
    assert cr["model_seed"] == tr["model_seed"]
    control_ids = {r["scene_id"] for r in cr["scene_knees"]
                   if r["eligible"] and r["tail_knee"] is not None}
    paired = [r for r in tr["scene_knees"] if r["eligible"]
              and r["tail_knee"] is not None and r["scene_id"] in control_ids]
    geometry = {r["scene_id"]: r for r in transfer["scene_geometry"]}
    cvs = {}
    for field in ("Dgeo", "Dfree", "Deuc"):
        ratios = [r["tail_knee"]/max(geometry[r["scene_id"]][field], 1) for r in paired]
        cvs[field] = st.pstdev(ratios)/st.mean(ratios)
    assert cvs["Dgeo"] < min(cvs["Dfree"], cvs["Deuc"])
    print("L paired ordering", cr["model_seed"], len(paired), cvs)

# Historical sensitivity checks: distinguish row recomputation from saved summaries.
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(EXP))
from bootstrap_ci import stats_of

boot = read("bootstrap_ci.json")
oc = aud["one_constant"]
point = stats_of({g: [r for r in eligible if r["grid"] == g] for g in (16,24,32)},
                 oc["geo_c"], oc["flat_T"])
for key, value in point.items():
    close(value, boot["point"][key])
assert boot["B"] == 2000
for f, expected in ((.9,(18.1,23.5)),(.95,(17.0,24.0)),(.99,(15.8,30.9))):
    ci = boot["ci"][f"saving_f{f}"]
    assert tuple(round(100*ci[k],1) for k in ("lo","hi")) == expected
print("bootstrap: all point estimates recomputed; CI rounding checked, replicates not rerun")

probe = read("probe_p100.json")
diffs, p100_c, cens100, cens95 = [], [], [], []
for g, gr in probe["per_grid"].items():
    for seed, run in gr["ckpts"].items():
        diffs.append(run["farcell"]["median_knee_joint"] - run["p95"]["median_knee_joint"])
        p100_c.append(run["p100"]["c1_median"])
        cens100.append(100*(1-run["p100"]["coverage"]))
        cens95.append(100*(1-run["p95"]["coverage"]))
        for metric in run.values():
            cv = metric["cv"]
            assert cv["dgeo"] < min(cv["dfree"],cv["deuc"])
assert min(diffs) == -1 and max(diffs) == 2
assert (round(min(p100_c),2),round(max(p100_c),2)) == (1.,1.33)
assert (round(min(cens100)),round(max(cens100))) == (7,47)
assert (round(min(cens95)),round(max(cens95))) == (2,11)
print("p100 saved summaries: joint median differences", diffs, "; not median paired delays")

sobel = read("results_M_sobel.json")
for g, gr in sobel["per_grid"].items():
    close(st.mean(gr["c1_geo"]["vals"]),gr["c1_geo"]["mean"])
    assert gr["c1_geo_cv"]["mean"] < min(gr["c1_free_cv"]["mean"],gr["c1_euc_cv"]["mean"])
    print("Sobel saved summaries",g,gr["ok_seeds"],gr["c1_geo"]["mean"],gr["c1_geo_cv"]["mean"])
assert sobel["per_grid"]["32"]["ok_seeds"] == [0,2]

n = read("branch_N_gates.json")
for knee in ("mean","tail"):
    wins = []
    for g in ("16","24","32"):
        gr = n["gate_R"][knee][g]
        for ruler in ("dgeo","dfree","deuc"):
            close(st.mean(v[ruler] for v in gr["per_seed"].values()),gr["seed_mean"][ruler])
        wins.append(sum(v["dgeo"] < min(v["dfree"],v["deuc"]) for v in gr["per_seed"].values()))
    assert wins == [3,3,2]
p = n["gate_P"]["pooled"]["tail"]
close(1-p["holdout_geo_cost"]/p["flat_T"],p["holdout_saving"])
assert round(100*p["holdout_saving"]) == -57
assert [round(100*v) for v in p["saving_ci"]] == [-72,-43]
print("N saved summaries: seed-mean ordering, 8/9 individual wins; tail saving",p["holdout_saving"])
print("PASS: selected row-based and saved-summary checks; not training/inference or full bootstrap replication")
