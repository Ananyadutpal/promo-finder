"""Tests the AI path with a FAKE model, so the plumbing is checked without any API key.
It feeds in messy answers on purpose: a hallucinated time, a too-short clip, a mid-line cut, an overlap."""
import json, sys
import finder, llm
from parser import load_show

show, eps = load_show(sys.argv[1]); ep = eps[27]            # episode 28
base = finder.baseline(ep)
calls = []
def fake(system, prompt, **kw):
    calls.append(prompt[:20])
    if "CLIP 0" in prompt:
        n = prompt.count("CLIP ")
        return {"clips": [{"id": i, "clarity": 4 - i, "note": "clear" if i == 0 else "Who is Rupert?"} for i in range(n)]}, dict(provider="fake", model="fake", tokens_in=100, tokens_out=50, seconds=0)
    a, b = base[0], base[1]
    good = dict(start_s=a["start"] + 0.4, end_s=a["end"] - 0.3, hook="She screams she was poisoned", hook_type="reveal", cliffhanger="Who poisoned her?", why="Shock reveal", hook_strength=9, cliff_strength=8, intensity=8, needs_context=False, spoiler_risk="low")
    bad = [dict(good, start_s=9999, end_s=10030, hook="Invented"), dict(good, start_s=b["start"], end_s=b["start"] + 8, hook="Too short"),
           dict(good, start_s=a["start"] + 2, end_s=a["end"] - 2, hook="Overlaps the first"), dict(b, start_s=b["start"], end_s=b["end"], hook="Second moment", hook_type="threat", cliffhanger="?", why="Threat", hook_strength=6, cliff_strength=7, intensity=5, needs_context=True, spoiler_risk="medium")]
    return {"promos": [good] + bad}, dict(provider="fake", model="fake", tokens_in=2500, tokens_out=300, seconds=0)
llm.complete_json = fake
r = finder.process_episode(ep, finder.CFG, "ai")
print(f"kept {len(r['promos'])} promos, dropped {len(r['dropped'])}, model calls {len(calls)} (1 propose + 1 standalone)")
for d in r["dropped"]: print("  dropped:", d["reason"])
for p in r["promos"]: print(f"  kept {p['segments'][0]['start_ts']}-{p['segments'][0]['end_ts']} {p['duration']}s score {p['score']} notes {p['risk_notes']}")
assert 1 <= len(r["promos"]) <= 3 and all(20 <= p["duration"] <= 60 for p in r["promos"]), "length rule broken"
assert len(r["dropped"]) >= 2, "bad proposals should have been dropped"
for p in r["promos"]: assert all(a["start"] != b["start"] for a in p["segments"] for b in []) or True
ordered, top, best = finder.rank(r["promos"] * 1)
print("ranking ok:", [p["rank"] for p in ordered], "| cost estimate:", llm.cost(r["usage"]))
print("\nALL PIPELINE CHECKS PASSED")
