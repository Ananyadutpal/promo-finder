"""Finds promo windows. Code enforces the rules; the AI only proposes, judges and explains."""
import re
import llm
from parser import clip_text, fmt, transcript

CFG = dict(min_len=20, max_len=60, per_ep=3, cap_top=2, top_n=10, hook_window=3.0, overlap=1.0,
           weights=dict(hook=0.30, cliff=0.30, clarity=0.25, intensity=0.15))
PROMPT_VERSION = "v1"

# ---------- boundaries: never start or end in the middle of a spoken line ----------
def _clean(ep, x):
    return not any(d["kind"] == "dialogue" and d["start"] + 0.05 < x < d["end"] - 0.05 for d in ep["events"])

def bounds(ep):
    st = sorted({e["start"] for e in ep["events"] if _clean(ep, e["start"])})
    en = sorted({e["end"] for e in ep["events"] if _clean(ep, e["end"])})
    return st, en

def fit(ep, s, e, cfg=CFG):
    """Snap a proposed window to clean boundaries and enforce the length rule. Returns (start, end) or None."""
    st, en = bounds(ep)
    if not st or not en: return None
    s = min(st, key=lambda v: abs(v - s))
    ok = [v for v in en if s + cfg["min_len"] <= v <= s + cfg["max_len"]]
    if not ok: return None
    e = min(ok, key=lambda v: abs(v - e))
    return (s, e)

def overlap(a, b): return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

def notes(ep, s, e, cfg=CFG):
    out = [] 
    first = [x for x in ep["events"] if s - 0.05 <= x["start"] <= s + cfg["hook_window"] and x["kind"] in ("dialogue", "action")]
    if not first: out.append("No speech or action in the first 3 seconds")
    if not any(x["kind"] == "dialogue" and s <= x["start"] < e for x in ep["events"]): out.append("No dialogue in the clip")
    if e >= ep["duration"] - 0.5: out.append("Runs to the end of the episode")
    return out

# ---------- baseline: rules only, no AI ----------
HOOK = re.compile(r"\b(slap|slaps|strike|strikes|gasp|gasps|scream|screams|blood|dagger|knife|kneel|kneels|betray\w*|secret|pregnant|poison\w*|dead|die|dies|kill\w*|how dare|who are you|divorce|duke|traitor|lie|lied|stop)\b", re.I)
STING = re.compile(r"cliffhanger|sting|stinger|swell|ominous|drone|boom|crash|silence", re.I)

def _score_window(ep, s, e, cfg):
    ev = [x for x in ep["events"] if s - 0.05 <= x["start"] < e]
    first = [x for x in ev if x["start"] <= s + cfg["hook_window"] and x["kind"] in ("dialogue", "action")]
    hook = min(10, sum(2 for x in first if x["kind"] == "dialogue" and re.search(r"[!?]", x["text"])) + sum(2 for x in first if HOOK.search(x["text"])) + (1 if first else 0))
    last = [x for x in ev if x["end"] >= e - 3.0]
    cliff = min(10, sum(3 for x in last if x["kind"] == "sound" and STING.search(x["text"])) + sum(2 for x in last if x["kind"] == "dialogue" and re.search(r"[?!]|\.\.\.$", x["text"])) + (2 if any("cut to black" in x["text"].lower() for x in last) else 0) + (1 if last else 0))
    inten = min(10, round(10 * sum(1 for x in ev if x["kind"] == "dialogue" and "!" in x["text"]) / max(1, (e - s) / 6)))
    return hook, cliff, inten

def baseline(ep, cfg=CFG):
    st, en = bounds(ep); cands = []
    for s in st:
        for e in en:
            if cfg["min_len"] <= e - s <= cfg["max_len"]:
                h, c, i = _score_window(ep, s, e, cfg); cands.append((h + c + 0.3 * i, s, e, h, c, i))
    cands.sort(reverse=True); keep = []
    for _, s, e, h, c, i in cands:
        if all(overlap((s, e), (k["start"], k["end"])) <= cfg["overlap"] for k in keep):
            keep.append(dict(start=s, end=e, hook_strength=h, cliff_strength=c, intensity=i, clarity=3, hook="(rule-based)", hook_type="rules",
                             cliffhanger="(rule-based)", why="Chosen by keyword and sound-cue rules, no AI.", needs_context=False, spoiler_risk="low"))
        if len(keep) >= cfg["per_ep"]: break
    return keep

# ---------- AI path ----------
SYSTEM = """You are a senior editor for short vertical drama apps (ReelShort, DramaBox). You choose promo clips that make a scrolling stranger watch the show."""
PROPOSE = """Below is the timed script of ONE episode (timestamps in seconds). Show logline: {logline}
Episode notes: {notes}

Pick up to {n} promo clips. A promo is ONE continuous stretch of {lo} to {hi} seconds taken from this episode.
Rules:
- HOOK: in the first 3 seconds something makes a scrolling viewer stop: a slap, a reveal, a shocking line, a threat, a confession.
- CLIFFHANGER: it ends at an unresolved moment, so the viewer must know what happens next. Cut BEFORE the answer, never after it.
- STANDALONE: someone who has never seen the show must follow it. Prefer moments whose key information is inside the clip.
- Do not give away the show's biggest twist. Do not overlap the clips. Use only timestamps that appear in the script.
Return JSON only: {{"promos":[{{"start_s":number,"end_s":number,"hook":"what happens in the first 3 seconds","hook_type":"slap|reveal|shocking line|threat|confession|other","cliffhanger":"what is left unresolved","why":"one sentence on why this works","hook_strength":1-10,"cliff_strength":1-10,"intensity":1-10,"needs_context":true|false,"spoiler_risk":"low|medium|high"}}]}}
Be honest: if the episode has fewer than {n} strong moments, return fewer.

SCRIPT
{script}"""
STANDALONE = """Each clip below is shown with NO other context. For each one, judge whether a viewer who has never seen the show would understand who is who, what is happening and what is at stake.
Return JSON only: {{"clips":[{{"id":number,"clarity":1-5,"note":"one short line on what is confusing or missing, or 'clear'"}}]}}

{clips}"""

def score(p, cfg=CFG):
    w = cfg["weights"]
    v = 100 * (w["hook"] * p["hook_strength"] / 10 + w["cliff"] * p["cliff_strength"] / 10 + w["clarity"] * p["clarity"] / 5 + w["intensity"] * p["intensity"] / 10)
    return round(v - {"low": 0, "medium": 4, "high": 10}.get(p.get("spoiler_risk", "low"), 0), 1)

def _num(x, lo, hi, d):
    try: return max(lo, min(hi, float(x)))
    except (TypeError, ValueError): return d

def process_episode(ep, cfg=CFG, mode="ai"):
    """Returns dict(promos=[...], raw={...}, usage=[...], dropped=[...])."""
    raw, usage, dropped = {}, [], []
    if mode == "baseline": props = baseline(ep, cfg)
    else:
        out, meta = llm.complete_json(SYSTEM, PROPOSE.format(logline=ep["logline"], notes=ep["continuity"], n=cfg["per_ep"], lo=cfg["min_len"], hi=cfg["max_len"], script=transcript(ep)))
        raw["proposal"], usage = out, usage + [meta]; props = []
        for p in (out.get("promos") or [])[:cfg["per_ep"] + 2]:
            f = fit(ep, _num(p.get("start_s"), 0, 1e6, -1), _num(p.get("end_s"), 0, 1e6, -1), cfg) if p.get("start_s") is not None else None
            if not f: dropped.append(dict(reason="timestamps could not be fitted to clean boundaries within the length rule", proposal=p)); continue
            props.append(dict(p, start=f[0], end=f[1], proposed=(p.get("start_s"), p.get("end_s")),
                              hook_strength=_num(p.get("hook_strength"), 1, 10, 5), cliff_strength=_num(p.get("cliff_strength"), 1, 10, 5),
                              intensity=_num(p.get("intensity"), 1, 10, 5), clarity=3))
        props.sort(key=lambda p: -(p["hook_strength"] + p["cliff_strength"])); keep = []
        for p in props:
            if all(overlap((p["start"], p["end"]), (k["start"], k["end"])) <= cfg["overlap"] for k in keep): keep.append(p)
            else: dropped.append(dict(reason="overlaps a stronger clip", proposal=p))
        props = keep[:cfg["per_ep"]]
        if props:
            clips = "\n\n".join(f"CLIP {i}\n{clip_text(ep, p['start'], p['end'])}" for i, p in enumerate(props))
            out2, meta2 = llm.complete_json(SYSTEM, STANDALONE.format(clips=clips)); raw["standalone"] = out2; usage.append(meta2)
            for c in out2.get("clips") or []:
                try: i = int(c["id"]); props[i]["clarity"] = _num(c.get("clarity"), 1, 5, 3); props[i]["clarity_note"] = c.get("note", "")
                except (KeyError, ValueError, IndexError, TypeError): pass
    promos = []
    for p in props:
        s, e = p["start"], p["end"]; n = notes(ep, s, e, cfg)
        if p.get("needs_context"): n.append("Model says it needs earlier context")
        if p.get("clarity_note") and p["clarity_note"].lower() != "clear": n.append("Standalone test: " + p["clarity_note"])
        if p.get("spoiler_risk") in ("medium", "high"): n.append(f"Spoiler risk: {p['spoiler_risk']}")
        promos.append(dict(episode=ep["episode"], segments=[dict(start=round(s, 1), end=round(e, 1), start_ts=fmt(s), end_ts=fmt(e))], duration=round(e - s, 1),
                           hook=p["hook"], hook_type=p.get("hook_type", ""), cliffhanger=p["cliffhanger"], why=p["why"], scores=dict(hook=p["hook_strength"], cliffhanger=p["cliff_strength"], clarity=p["clarity"], intensity=p["intensity"]),
                           score=score(p, cfg), risk_notes=n, script=clip_text(ep, s, e), mode=mode))
    return dict(promos=promos, raw=raw, usage=usage, dropped=dropped)

def rank(all_promos, cfg=CFG):
    """Global ranking, best per episode, and the 'post first' list (capped per episode so one episode cannot dominate)."""
    ordered = sorted(all_promos, key=lambda p: -p["score"])
    for i, p in enumerate(ordered, 1): p["rank"] = i
    top, per = [], {}
    for p in ordered:
        if per.get(p["episode"], 0) < cfg["cap_top"] and len(top) < cfg["top_n"]: top.append(p); per[p["episode"]] = per.get(p["episode"], 0) + 1
    best = {}
    for p in ordered: best.setdefault(p["episode"], p)
    return ordered, top, best
