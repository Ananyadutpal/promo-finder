"""python list_models.py  -> shows which Gemini models your key can use."""
import llm, os
key = os.getenv("GEMINI_API_KEY")
r = llm.requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", headers={"x-goog-api-key": key or ""}, timeout=30)
print("Status", r.status_code)
for m in r.json().get("models", []):
    if "generateContent" in m.get("supportedGenerationMethods", []): print(" ", m["name"].split("/")[-1])
try: print("\nThe code would choose:", llm.pick_gemini_model(key))
except Exception as e: print("Problem:", e)