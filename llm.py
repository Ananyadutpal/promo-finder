"""One function the rest of the code calls. The model is a setting, so it is easy to swap."""
import json, os, pathlib, re, time
import requests

def _env():
    f = pathlib.Path(__file__).with_name(".env")
    if f.exists():
        for ln in f.read_text().splitlines():
            if "=" in ln and not ln.strip().startswith("#"):
                k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
_env()

DEFAULTS = {"gemini": "gemini-3.8-flash", "openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-5"}
PRICE = {"in": float(os.getenv("PRICE_IN_PER_M", "0.30")), "out": float(os.getenv("PRICE_OUT_PER_M", "2.50"))}   # USD per 1M tokens: set the real list price of your model
_last = [0.0]

class LLMError(Exception): pass

def settings():
    p = os.getenv("LLM_PROVIDER", "gemini")
    return dict(provider=p, model=os.getenv("LLM_MODEL", DEFAULTS.get(p, "")))

def _json(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try: return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a: return json.loads(t[a:b + 1])
        raise LLMError("The model did not return valid JSON")

def _post(url, headers, body):
    timeout, r, last = float(os.getenv("LLM_TIMEOUT", "240")), None, None
    for wait in (0, 5, 15, 40):
        time.sleep(wait)
        try: r = requests.post(url, headers=headers, json=body, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e: last, r = e, None; continue
        if r.status_code not in (429, 500, 503) and "json_validate_failed" not in r.text: break
    if r is None: raise LLMError(f"No answer from the model service ({type(last).__name__}). Check your internet connection, or try again.")
    if r.status_code != 200: raise LLMError(f"{r.status_code}: {r.text[:300]}")
    return r.json()

def complete_json(system, prompt, provider=None, model=None):
    """Returns (parsed_json, meta). meta has tokens and seconds, used for the cost estimate."""
    s = settings(); provider, model = provider or s["provider"], model or s["model"]
    gap = float(os.getenv("LLM_MIN_INTERVAL", "6")) - (time.time() - _last[0])      # stay under free-tier rate limits
    if gap > 0: time.sleep(gap)
    t = time.time()
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY") or _fail("GEMINI_API_KEY is not set")
        cfg = {"temperature": 0.2, "responseMimeType": "application/json", "maxOutputTokens": 8192}
        if "2.5" in model: cfg["thinkingConfig"] = {"thinkingBudget": 0}
        elif re.search(r"gemini-[3-9]", model): cfg["thinkingConfig"] = {"thinkingLevel": os.getenv("LLM_THINKING", "low")}    # long "thinking" makes simple tasks slow
        url = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
        body = lambda c: {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"parts": [{"text": prompt}]}], "generationConfig": c}
        print(f"    sending ~{len(prompt) // 4} tokens to {model}...", flush=True)
        try: d = _post(url.format(model), {"x-goog-api-key": key}, body(cfg))
        except LLMError as e:
            msg = str(e)
            if msg.startswith("400") and "hink" in msg and "thinkingConfig" in cfg:      # this model does not accept that setting
                cfg.pop("thinkingConfig"); d = _post(url.format(model), {"x-goog-api-key": key}, body(cfg))
            elif msg.startswith("404") and not os.getenv("LLM_MODEL"):                  # an explicitly chosen model is never replaced silently
                model = pick_gemini_model(key); os.environ["LLM_MODEL"] = model; print(f"  (model unavailable, switched to {model})")
                d = _post(url.format(model), {"x-goog-api-key": key}, body(cfg))
            else: raise
        try: text = d["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError): raise LLMError("Gemini returned no text")
        u = d.get("usageMetadata", {}); tin, tout = u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
    elif provider == "openai":      # any OpenAI-compatible API: set OPENAI_BASE_URL for other hosts
        key = os.getenv("OPENAI_API_KEY") or _fail("OPENAI_API_KEY is not set")
        body = {"model": model, "temperature": 0.2, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if "gpt-oss" in model: body.update(reasoning_effort="low", max_completion_tokens=8000)
        d = _post(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions", {"Authorization": f"Bearer {key}"}, body)
        text = d["choices"][0]["message"]["content"]; u = d.get("usage", {}); tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    elif provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY") or _fail("ANTHROPIC_API_KEY is not set")
        d = _post("https://api.anthropic.com/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"},
                  {"model": model, "max_tokens": 4000, "temperature": 0.2, "system": system, "messages": [{"role": "user", "content": prompt + "\n\nReturn JSON only."}]})
        text = "".join(b.get("text", "") for b in d["content"]); u = d.get("usage", {}); tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
    else: raise LLMError(f"Unknown provider {provider}")
    _last[0] = time.time(); print(f"    answered in {time.time() - t:.1f}s ({tout} tokens out)", flush=True)
    return _json(text), dict(provider=provider, model=model, tokens_in=tin, tokens_out=tout, seconds=round(time.time() - t, 1))

def _fail(m): raise LLMError(m)

def pick_gemini_model(key):
    """Ask Google which models this key can use, and choose the newest Flash model."""
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", headers={"x-goog-api-key": key}, timeout=30)
    if r.status_code != 200: raise LLMError(f"Could not list models: {r.status_code} {r.text[:200]}")
    best = None
    for m in r.json().get("models", []):
        name = m["name"].split("/")[-1]; v = re.match(r"gemini-(\d+(?:\.\d+)?)-flash(-lite)?$", name)
        if v and "generateContent" in m.get("supportedGenerationMethods", []):
            rank = (float(v.group(1)), 0 if v.group(2) else 1)           # newest version first, full Flash before Flash-Lite
            if not best or rank > best[0]: best = (rank, name)
    if not best: raise LLMError("No Gemini Flash model is available for this key")
    return best[1]

def cost(usage):
    tin, tout = sum(u["tokens_in"] for u in usage), sum(u["tokens_out"] for u in usage)
    return dict(tokens_in=tin, tokens_out=tout, usd=round(tin / 1e6 * PRICE["in"] + tout / 1e6 * PRICE["out"], 4))