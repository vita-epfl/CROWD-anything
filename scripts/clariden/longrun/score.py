"""Scores of a long-horizon run folder per window start (python3.11 score.py <run>): prompt following per side
(evaluation.prompt_following), topology per attribute (Cohen's kappa of each side against OSM, and agreement of each
generated side with the real clip), and vehicle speeds against the OSM limit. Writes <run>/scores_*.csv.
Run it in the wm_data_manifest container (numpy)."""
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/iopsstor/scratch/cscs/malam/rgwm-eval/CROWD-anything")
from evaluation.prompt_following import read_batch_answers as pf_answers, read_statements, summarize  # noqa: E402
from evaluation.topology import read_batch_answers as topo_answers  # noqa: E402

run = Path(sys.argv[1])
start = {r["window"]: r["start_s"] for r in csv.DictReader(open(run / "windows.csv"))}


def write(rows, name):
    if rows:
        with open(run / name, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("\n" + name)
        for r in rows:
            print("  " + "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


# prompt following
statements = read_statements(run / "statements.json")
rows = pf_answers(sorted((run / "batch_pf").rglob("answers-*.json")), statements)
out = []
for s in sorted({start[r["clip_id"]] for r in rows}, key=float):
    for summary in summarize([r for r in rows if start[r["clip_id"]] == s]):
        keys = ("side", "clips", "confirmed", "rejected", "balanced_on_real", "unanswered")
        out.append({"window_s": s, **{k: summary[k] for k in keys}})
write(out, "scores_prompt_following.csv")

# topology
facts = json.load(open(run / "facts.json"))
answers = topo_answers(sorted((run / "batch_topo").rglob("answers-*.json")), facts, "qwen-drive")
by = collections.defaultdict(list)
for a in answers:
    side = a.get("side", "reference")
    by[(start[a["clip_id"]], side, a["attribute"])].append(a)
out = []
for (s, side, attribute), rs in sorted(by.items(), key=lambda k: (float(k[0][0]), k[0][1], k[0][2])):
    ok = [a for a in rs if a["answer"] is not None]
    if not ok:
        continue
    acc = sum(a["answer"] == a["truth"] for a in ok) / len(ok)
    pt = collections.Counter(a["truth"] for a in ok)
    pa = collections.Counter(a["answer"] for a in ok)
    pe = sum(pt[v] * pa[v] for v in set(pt) | set(pa)) / len(ok) ** 2
    out.append({"window_s": s, "side": side, "attribute": attribute, "n": len(rs), "answered": len(ok),
                "accuracy": acc, "kappa": (acc - pe) / (1 - pe) if pe < 1 else 0.0})
write(out, "scores_topology.csv")

# vehicle speed
per_video = run / "results_speed" / "per_video.csv"
if per_video.exists():
    out = []
    videos = list(csv.DictReader(open(per_video)))
    for s in sorted({start[v["scene"]] for v in videos}, key=float):
        for side in sorted({v["side"] for v in videos}):
            vs = [v for v in videos if start[v["scene"]] == s and v["side"] == side]
            moving = [float(v["median_moving_kmh"]) for v in vs if v["median_moving_kmh"]]
            ratios = [float(v["median_moving_kmh"]) / float(v["limit_kmh"]) for v in vs
                      if v["median_moving_kmh"] and v["limit_kmh"]]
            n_moving = sum(int(v["moving"]) for v in vs if v["limit_kmh"])
            above = sum(int(v["above_limit"] or 0) for v in vs if v["limit_kmh"])
            out.append({"window_s": s, "side": side, "videos": len(vs), "with_moving": len(moving),
                        "median_moving_kmh": sorted(moving)[len(moving) // 2] if moving else "",
                        "speed_over_limit": sorted(ratios)[len(ratios) // 2] if ratios else "",
                        "share_above_limit": above / n_moving if n_moving else ""})
    write(out, "scores_speed.csv")
