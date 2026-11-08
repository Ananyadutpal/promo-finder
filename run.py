"""python run.py <zip|folder|file> [--mode ai|baseline] [--limit N] [--out outputs]"""
import argparse, csv, hashlib, json, pathlib, sys, time
import finder, llm
from parser import load_show

def episode_key(ep, mode, cfg):
    s = {"provider": "openai", "model": "openai/gpt-oss-120b"} if mode == "ai" else ""
    raw = json.dumps([ep["events"], mode, s, finder.PROMPT_VERSION, cfg["min_len"], cfg["max_len"], cfg["per_ep"]], default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]
 
def run_show(path, mode="ai", limit=None, out="outputs", cfg=finder.CFG, log=print, cache_dir="cache"):
    show, eps = load_show(path); eps = eps[:limit] if limit else eps
    outdir = pathlib.Path(out) / (show.replace(" ", "_")[:60] + f"_{mode}"); (outdir / "raw").mkdir(parents=True, exist_ok=True); pathlib.Path(cache_dir).mkdir(exist_ok=True)
    allp, usage, errors, cached = [], [], [], 0
    for ep in eps:
        f = pathlib.Path(cache_dir) / f"{episode_key(ep, mode, cfg)}.json"
        try:
            if f.exists(): res, cached = json.loads(f.read_text()), cached + 1
            else: res = finder.process_episode(ep, cfg, mode); f.write_text(json.dumps(res))
        except Exception as e:
            errors.append(dict(episode=ep["episode"], error=str(e))); log(f"  episode {ep['episode']}: FAILED ({e})"); continue
        (outdir / "raw" / f"episode_{ep['episode']:03d}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        allp += res["promos"]; usage += res["usage"]; log(f"  episode {ep['episode']:>3}: {len(res['promos'])} promos" + (f", {len(res['dropped'])} dropped" if res["dropped"] else ""))
    ordered, top, best = finder.rank(allp, cfg)
    (outdir / "promos.json").write_text(json.dumps(ordered, indent=1, ensure_ascii=False))
    with open(outdir / "promos.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["rank", "episode", "start", "end", "duration_s", "score", "hook", "cliffhanger", "why", "risk_notes"])
        for p in ordered:
            sg = p["segments"][0]; w.writerow([p["rank"], p["episode"], sg["start_ts"], sg["end_ts"], p["duration"], p["score"], p["hook"], p["cliffhanger"], p["why"], "; ".join(p["risk_notes"])])
    c = llm.cost(usage)
    md = [f"# Post-first list: {show} ({mode})", ""] + [f"{i}. **Episode {p['episode']}, {p['segments'][0]['start_ts']} to {p['segments'][0]['end_ts']}** ({p['duration']}s, score {p['score']}). Hook: {p['hook']} Cliffhanger: {p['cliffhanger']} Why: {p['why']}" for i, p in enumerate(top, 1)]
    (outdir / "top10.md").write_text("\n".join(md), encoding="utf-8")
    summary = dict(show=show, mode=mode, episodes=len(eps), promos=len(ordered), from_cache=cached, errors=errors, cost=c, model=llm.settings() if mode == "ai" else None)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1)); return dict(summary=summary, ordered=ordered, top=top, best=best, outdir=str(outdir))

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("path"); a.add_argument("--mode", default="ai"); a.add_argument("--limit", type=int); a.add_argument("--out", default="outputs"); a = a.parse_args()
    r = run_show(a.path, a.mode, a.limit, a.out); print(json.dumps(r["summary"], indent=1)); print("Outputs in", r["outdir"])
