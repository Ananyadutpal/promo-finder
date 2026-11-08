"""Web UI for the promo finder.  Run locally: uvicorn app:app --reload   |  Render: uvicorn app:app --host 0.0.0.0 --port $PORT"""
import json, pathlib, shutil, tempfile, threading, uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import llm
from run import run_show

app = FastAPI(title="Promo Finder")
JOBS = {}
WORK = pathlib.Path(tempfile.gettempdir()) / "promo_jobs"; WORK.mkdir(exist_ok=True)


def worker(jid, path, mode, limit):
    job = JOBS[jid]
    def log(m): job["log"].append(str(m))
    try:
        r = run_show(path, mode, limit, out=str(WORK / jid / "out"), log=log, cache_dir="cache")
        job.update(status="done", result=dict(summary=r["summary"], ordered=r["ordered"], top=r["top"]), outdir=r["outdir"])
    except Exception as e:
        job.update(status="error", error=str(e))


@app.post("/api/run")
async def start(file: UploadFile = File(...), mode: str = Form("ai"), limit: int = Form(0)):
    jid = uuid.uuid4().hex[:10]
    d = WORK / jid; d.mkdir(parents=True, exist_ok=True)
    path = d / (file.filename or "upload.zip")
    with open(path, "wb") as fh: shutil.copyfileobj(file.file, fh)
    JOBS[jid] = dict(status="running", log=[], model=llm.settings() if mode == "ai" else None)
    threading.Thread(target=worker, args=(jid, str(path), mode, limit or None), daemon=True).start()
    return dict(id=jid)


@app.get("/api/job/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j: raise HTTPException(404, "unknown job")
    return dict(status=j["status"], log=j["log"][-40:], error=j.get("error"), result=j.get("result"), model=j.get("model"))


@app.get("/api/download/{jid}/{kind}")
def download(jid: str, kind: str):
    j = JOBS.get(jid)
    if not j or "outdir" not in j or kind not in ("csv", "json"): raise HTTPException(404)
    return FileResponse(pathlib.Path(j["outdir"]) / f"promos.{kind}", filename=f"promos.{kind}")


@app.get("/api/health")
def health(): return dict(ok=True, model=llm.settings())


@app.get("/", response_class=HTMLResponse)
def index(): return HTML


HTML = r"""<!doctype html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Promo Finder</title>
<style>
:root{--bg:#0d0f14;--card:#161a22;--bd:#262c38;--tx:#e8ebf2;--mu:#8b93a7;--ac:#7c5cff;--ac2:#b794ff;--ok:#2fbf71;--wn:#f0a93b;--bad:#ef5b5b}
[data-theme=light]{--bg:#f5f6fa;--card:#fff;--bd:#e2e5ee;--tx:#171a22;--mu:#667085;--ac:#6b46f0;--ac2:#5a36d6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:24px 16px 80px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}
h1{font-size:22px;margin:0}h1 span{color:var(--ac2)}
button,.btn{background:var(--ac);color:#fff;border:0;border-radius:10px;padding:10px 16px;font-weight:600;cursor:pointer;text-decoration:none;font-size:14px}
button.ghost,.btn.ghost{background:transparent;color:var(--tx);border:1px solid var(--bd)}
button:disabled{opacity:.5;cursor:wait}
.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:18px;margin-bottom:14px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
input[type=file],select,input[type=number]{background:var(--bg);color:var(--tx);border:1px solid var(--bd);border-radius:10px;padding:9px}
input[type=number]{width:90px}
label{color:var(--mu);font-size:13px}
.drop{border:2px dashed var(--bd);border-radius:12px;padding:22px;text-align:center;color:var(--mu);margin-bottom:12px}
.log{font:12px/1.5 ui-monospace,Consolas,monospace;color:var(--mu);max-height:150px;overflow:auto;white-space:pre-wrap}
.bar{height:6px;background:var(--bd);border-radius:6px;overflow:hidden;margin:10px 0}.bar i{display:block;height:100%;width:30%;background:var(--ac);animation:mv 1.2s infinite ease-in-out}
@keyframes mv{0%{margin-left:-30%}100%{margin-left:100%}}
.tabs{display:flex;gap:8px;margin:18px 0 12px}.tab{padding:8px 14px;border-radius:999px;border:1px solid var(--bd);cursor:pointer;color:var(--mu)}.tab.on{background:var(--ac);color:#fff;border-color:var(--ac)}
.rank{font-size:26px;font-weight:800;color:var(--ac2);min-width:44px}
.head{display:flex;gap:14px;align-items:flex-start}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 8px}
.pill{font-size:12px;padding:2px 9px;border-radius:999px;border:1px solid var(--bd);color:var(--mu)}
.pill.sc{color:#fff;background:var(--ok);border-color:var(--ok)}
.k{color:var(--mu);font-size:12px;text-transform:uppercase;letter-spacing:.05em;margin-top:8px}
.warn{color:var(--wn);font-size:13px;margin-top:8px}
details summary{cursor:pointer;color:var(--ac2);margin-top:10px;font-size:13px}
pre{white-space:pre-wrap;background:var(--bg);border:1px solid var(--bd);border-radius:10px;padding:10px;font-size:12px}
.stats{display:flex;gap:10px;flex-wrap:wrap}.stat{flex:1;min-width:120px;background:var(--bg);border:1px solid var(--bd);border-radius:10px;padding:10px}.stat b{display:block;font-size:20px}
.err{color:var(--bad)}
</style></head><body><div class="wrap">
<header><h1>Promo <span>Finder</span></h1><button class="ghost" id="th">Light / Dark</button></header>
<div class="card">
  <div class="drop">Choose a screenplay <b>.zip</b> or a single <b>.txt</b> episode</div>
  <div class="row">
    <input type="file" id="f" accept=".zip,.txt">
    <label>Mode <select id="m"><option value="ai">AI</option><option value="baseline">Rules only (fast)</option></select></label>
    <label>Episodes (0 = all) <input type="number" id="l" value="0" min="0"></label>
    <button id="go">Find promos</button>
  </div>
  <div id="st" style="margin-top:12px"></div>
</div>
<div id="out"></div>
</div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const root=document.documentElement;root.dataset.theme=localStorage.getItem('t')||(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark');
$('th').onclick=()=>{root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';try{localStorage.setItem('t',root.dataset.theme)}catch(e){}};
let jid,data,tab='top';
$('go').onclick=async()=>{
  const f=$('f').files[0];if(!f){$('st').innerHTML='<span class=err>Pick a file first.</span>';return}
  const fd=new FormData();fd.append('file',f);fd.append('mode',$('m').value);fd.append('limit',$('l').value||0);
  $('go').disabled=true;$('out').innerHTML='';$('st').innerHTML='Uploading...';
  const r=await fetch('/api/run',{method:'POST',body:fd});const j=await r.json();jid=j.id;poll();
};
async function poll(){
  const r=await fetch('/api/job/'+jid);const j=await r.json();
  if(j.status==='running'){$('st').innerHTML='<div class="bar"><i></i></div><div class="log">'+esc(j.log.join('\n'))+'</div>';setTimeout(poll,2000)}
  else if(j.status==='error'){$('st').innerHTML='<span class=err>Failed: '+esc(j.error)+'</span>';$('go').disabled=false}
  else{$('st').innerHTML='Done.';$('go').disabled=false;data=j.result;render()}
}
function card(p){const s=p.segments[0];return `<div class="card"><div class="head"><div class="rank">#${p.rank}</div><div style="flex:1">
<div class="meta"><span class="pill sc">score ${p.score}</span><span class="pill">Episode ${p.episode}</span><span class="pill">${s.start_ts} - ${s.end_ts}</span><span class="pill">${p.duration}s</span><span class="pill">${esc(p.hook_type)}</span></div>
<div class="k">Hook</div>${esc(p.hook)}<div class="k">Cliffhanger</div>${esc(p.cliffhanger)}<div class="k">Why it works</div>${esc(p.why)}
${p.risk_notes.length?'<div class="warn">&#9888; '+p.risk_notes.map(esc).join(' &middot; ')+'</div>':''}
<details><summary>Script</summary><pre>${esc(p.script)}</pre></details></div></div></div>`}
function render(){
  const sm=data.summary,list=tab==='top'?data.top:data.ordered;
  $('out').innerHTML=`<div class="card"><div class="stats"><div class="stat"><b>${sm.episodes}</b>episodes</div><div class="stat"><b>${sm.promos}</b>promos</div><div class="stat"><b>$${sm.cost.usd}</b>est. cost</div><div class="stat"><b>${sm.errors.length}</b>failed episodes</div></div>
<div class="row" style="margin-top:12px"><a class="btn ghost" href="/api/download/${jid}/csv">Download CSV</a><a class="btn ghost" href="/api/download/${jid}/json">Download JSON</a></div></div>
<div class="tabs"><div class="tab ${tab==='top'?'on':''}" onclick="tab='top';render()">Post first (top ${data.top.length})</div><div class="tab ${tab==='all'?'on':''}" onclick="tab='all';render()">All promos (${data.ordered.length})</div></div>`+list.map(card).join('');
}
</script></body></html>"""