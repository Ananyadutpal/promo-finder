"""Turns Kuku-style screenplay files into timed events. Plain code, no AI."""
import io, pathlib, re, zipfile

RANGE = re.compile(r"Start:\s*\(([\d:.]+)\)\s*\|\s*End:\s*\(([\d:.]+)\)")
LABEL = re.compile(r"^\*\*(.+?)\s*:\*\*\s*$")
KINDS = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound", "SFX": "sound"}

def secs(t):
    s = 0.0
    for p in t.split(":"): s = s * 60 + float(p)
    return s

def fmt(t):
    m, s = divmod(max(t, 0), 60); return f"{int(m):02d}:{s:04.1f}"

def parse_episode(text, name=""):
    g = lambda pat: (re.search(pat, text) or [None, ""])[1].strip()
    ep = dict(name=name, episode=None, logline=g(r"EPISODE LOGLINE:\s*(.+)"), continuity=g(r"CONTINUITY NOTES:\s*(.+)"), events=[])
    m = re.search(r"EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", text)
    ep["episode"] = int(m.group(1)) if m else None
    lines, heading, i = text.splitlines(), "", 0
    while i < len(lines):
        ln = lines[i].strip()
        h = re.match(r"Scene heading:\s*(.+)", ln)
        if h: heading = h.group(1)
        lab = LABEL.match(ln)
        rg = RANGE.search(lines[i + 1]) if lab and i + 1 < len(lines) else None
        if rg:
            j, body = i + 2, []
            while j < len(lines) and lines[j].strip() and not LABEL.match(lines[j].strip()): body.append(lines[j].strip()); j += 1
            label = lab.group(1).strip(); kind = KINDS.get(label.upper(), "dialogue")
            who = re.sub(r"\s*\(.*", "", label).strip() if kind == "dialogue" else ""
            tone = " ".join(re.findall(r"\(([^)]*)\)", label)) if kind == "dialogue" else ""
            ep["events"].append(dict(kind=kind, who=who, tone=tone, start=secs(rg.group(1)), end=secs(rg.group(2)),
                                     text=" ".join(body).strip().strip('"'), scene=heading))
            i = j; continue
        i += 1
    order = {"dialogue": 0, "action": 1, "camera": 2, "sound": 3}
    ep["events"].sort(key=lambda e: (e["start"], order[e["kind"]]))
    for n, e in enumerate(ep["events"]): e["i"] = n
    ep["duration"] = max((e["end"] for e in ep["events"]), default=0.0)
    return ep

def load_show(path):
    """A .zip, a folder or a single .txt. Returns (show_name, episodes sorted by episode number)."""
    p, texts = pathlib.Path(path), []
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            texts = [(pathlib.Path(n).stem, z.read(n).decode("utf-8", "ignore")) for n in sorted(z.namelist()) if n.lower().endswith(".txt") and "__MACOSX" not in n]
        show = p.stem
    elif p.is_dir():
        texts = [(f.stem, f.read_text("utf-8", "ignore")) for f in sorted(p.rglob("*.txt"))]; show = p.name
    else:
        texts = [(p.stem, p.read_text("utf-8", "ignore"))]; show = p.stem
    eps = [parse_episode(t, n) for n, t in texts]
    eps = [e for e in eps if e["events"]]
    for k, e in enumerate(sorted(eps, key=lambda e: (e["episode"] is None, e["episode"] or 0, e["name"]))): e["episode"] = e["episode"] or k + 1
    return show, sorted(eps, key=lambda e: e["episode"])

def transcript(ep):
    out = []
    for e in ep["events"]:
        who = {"dialogue": f"{e['who']}" + (f" ({e['tone']})" if e["tone"] else ""), "action": "ACTION", "camera": "CAMERA", "sound": "SOUND"}[e["kind"]]
        out.append(f"[{e['start']:.1f}-{e['end']:.1f}] {who}: {e['text']}")
    return "\n".join(out)

def clip_text(ep, s, e, with_camera=True):
    ev = [x for x in ep["events"] if x["start"] >= s - 0.05 and x["start"] < e - 0.05 and (with_camera or x["kind"] != "camera")]
    return "\n".join(f"[{x['start'] - s:.1f}s] " + (f"{x['who']}: " if x["kind"] == "dialogue" else x["kind"].upper() + ": ") + x["text"] for x in ev)
