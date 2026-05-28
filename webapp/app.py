#!/usr/bin/env python3
"""Red-Green-Dragon CUDA-to-HIP transpiler web app."""

import io
import queue
import threading
import uuid
import zipfile
from typing import List

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

# ── Model ─────────────────────────────────────────────────────────────────────

_state: dict = {"model": None, "tokenizer": None, "ready": False, "error": None}


def _load_model() -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading red-green-dragon-src-7b from local weights…")
        _state["model"] = AutoModelForCausalLM.from_pretrained(
            "./models/red-green-dragon-src-7b", torch_dtype="auto", device_map="auto"
        )
        _state["tokenizer"] = AutoTokenizer.from_pretrained("./models/red-green-dragon-src-7b")
        _state["ready"] = True
        print("Model ready.")
    except Exception as exc:
        _state["error"] = str(exc)
        print(f"Model load failed: {exc}")


threading.Thread(target=_load_model, daemon=True).start()

# ── Jobs ──────────────────────────────────────────────────────────────────────

_jobs: dict = {}
_job_queue: queue.Queue = queue.Queue()


def _transpile(cuda_code: str) -> str:
    model = _state["model"]
    tok = _state["tokenizer"]
    prompt = f"Convert the following CUDA code to AMD GPU code:\n```cuda\n{cuda_code}\n```"
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=4096)
    response = tok.batch_decode(
        [ids[len(inp) :] for inp, ids in zip(inputs.input_ids, out_ids)],
        skip_special_tokens=True,
    )[0]
    hip = response.split("```amd")[-1].split("```")[0]
    return hip.strip() if hip.strip() else response.strip()


def _worker() -> None:
    while True:
        job_id = _job_queue.get()
        job = _jobs[job_id]
        job["status"] = "processing"
        for entry in job["files"]:
            job["current"] = entry["name"]
            try:
                hip = _transpile(entry["code"])
                job["results"][entry["name"]] = {
                    "ok": True,
                    "hip": hip,
                    "cuda": entry["code"],
                    "rel_path": entry["rel_path"],
                }
            except Exception as exc:
                job["results"][entry["name"]] = {
                    "ok": False,
                    "error": str(exc),
                    "cuda": entry["code"],
                    "rel_path": entry["rel_path"],
                }
            job["done"] += 1
        job["status"] = "done"
        job["current"] = None
        _job_queue.task_done()


threading.Thread(target=_worker, daemon=True).start()

# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI(title="Red-Green-Dragon Transpiler")


@app.get("/api/status")
def api_status() -> dict:
    return {"ready": _state["ready"], "error": _state["error"]}


@app.post("/api/jobs")
async def create_job(files: List[UploadFile] = File(...)) -> dict:
    if not _state["ready"]:
        raise HTTPException(503, _state["error"] or "Model is still loading")

    entries = []
    for f in files:
        if not (f.filename.endswith(".cu") or f.filename.endswith(".cuh")):
            continue
        code = (await f.read()).decode("utf-8", errors="replace")
        entries.append({"name": f.filename, "rel_path": f.filename, "code": code})

    if not entries:
        raise HTTPException(400, "No .cu or .cuh files found in the upload")

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {
        "status": "queued",
        "files": entries,
        "results": {},
        "current": None,
        "done": 0,
        "total": len(entries),
    }
    _job_queue.put(job_id)
    return {"job_id": job_id, "total": len(entries)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status": job["status"],
        "current": job["current"],
        "done": job["done"],
        "total": job["total"],
        "results": job["results"],
    }


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str) -> StreamingResponse:
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Job not ready for download")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, res in job["results"].items():
            if res["ok"]:
                out_name = res["rel_path"].replace(".cu", ".hip").replace(".cuh", "_hip.h")
                zf.writestr(out_name, res["hip"])
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="red-green-dragon_{job_id}.zip"'},
    )


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(_HTML)


# ── HTML ──────────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Red-Green-Dragon — CUDA → HIP Transpiler</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/styles/github-dark.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/languages/cpp.min.js"></script>
<style>
  .hljs { background: transparent !important; }
  pre { margin: 0; }
  .drop-active { border-color: #3b82f6 !important; background: rgba(59,130,246,.06); }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
</style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen">

<!-- Header -->
<header class="bg-gray-900 border-b border-gray-800 px-8 py-4 sticky top-0 z-10">
  <div class="max-w-6xl mx-auto flex items-center justify-between">
    <div>
      <span class="text-lg font-bold tracking-tight">Red-Green-Dragon</span>
      <span class="ml-2 text-gray-500 text-sm">CUDA → HIP Transpiler</span>
    </div>
    <div class="flex items-center gap-2 text-sm">
      <span id="status-dot" class="w-2 h-2 rounded-full bg-yellow-400"></span>
      <span id="status-text" class="text-gray-400">Loading model…</span>
    </div>
  </div>
</header>

<main class="max-w-6xl mx-auto px-8 py-10 space-y-8">

  <!-- Upload section -->
  <section id="upload-section">
    <div id="drop-zone"
         class="border-2 border-dashed border-gray-700 rounded-2xl p-16 text-center cursor-pointer transition-all hover:border-gray-600"
         ondragover="event.preventDefault(); this.classList.add('drop-active')"
         ondragleave="this.classList.remove('drop-active')"
         ondrop="handleDrop(event); this.classList.remove('drop-active')"
         onclick="fileInput.click()">
      <div class="text-5xl mb-4">🔄</div>
      <p class="text-lg text-gray-300">
        Drop <code class="bg-gray-800 px-1.5 py-0.5 rounded text-sm">.cu</code>
        / <code class="bg-gray-800 px-1.5 py-0.5 rounded text-sm">.cuh</code> files here
      </p>
      <p class="text-sm text-gray-600 mt-1">or choose with the buttons below</p>
      <div class="mt-6 flex justify-center gap-3" onclick="event.stopPropagation()">
        <button onclick="fileInput.click()"
                class="px-4 py-2 bg-blue-700 hover:bg-blue-600 rounded-lg text-sm font-medium transition-colors">
          Select Files
        </button>
        <button onclick="folderInput.click()"
                class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm font-medium transition-colors">
          Select Folder
        </button>
      </div>
    </div>

    <input id="fileInput" type="file" multiple accept=".cu,.cuh" class="hidden" onchange="handleFiles(this.files)"/>
    <input id="folderInput" type="file" webkitdirectory class="hidden" onchange="handleFiles(this.files)"/>

    <div id="file-list" class="mt-4 hidden">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm text-gray-400"><span id="file-count">0</span> file(s) selected</span>
        <button onclick="clearFiles()" class="text-xs text-gray-600 hover:text-red-400 transition-colors">Clear all</button>
      </div>
      <div id="file-items" class="space-y-1 max-h-52 overflow-y-auto pr-1"></div>
      <div class="mt-4 flex justify-end">
        <button id="run-btn" onclick="run()"
                class="px-6 py-2.5 bg-green-600 hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors">
          Transpile →
        </button>
      </div>
    </div>
  </section>

  <!-- Progress section -->
  <section id="progress-section" class="hidden">
    <div class="bg-gray-900 rounded-2xl border border-gray-800 p-6 space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="font-semibold text-gray-200">Transpiling…</h2>
        <span id="progress-fraction" class="text-sm text-gray-400 tabular-nums">0 / 0</span>
      </div>
      <div class="w-full bg-gray-800 rounded-full h-1.5">
        <div id="progress-bar" class="bg-blue-500 h-1.5 rounded-full transition-all duration-500" style="width:0"></div>
      </div>
      <p id="progress-file" class="text-sm text-blue-400 truncate h-5"></p>
    </div>
  </section>

  <!-- Results section -->
  <section id="results-section" class="hidden space-y-6">
    <div class="flex items-center justify-between">
      <h2 class="text-lg font-semibold">Results</h2>
      <div class="flex gap-3">
        <button onclick="reset()"
                class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition-colors">
          New Job
        </button>
        <button onclick="downloadZip()"
                class="px-4 py-2 bg-green-600 hover:bg-green-500 rounded-lg text-sm font-semibold transition-colors">
          ⬇ Download ZIP
        </button>
      </div>
    </div>
    <div id="results-list" class="space-y-4"></div>
  </section>

</main>

<script>
const fileInput  = document.getElementById('fileInput');
const folderInput = document.getElementById('folderInput');
let selected = [];
let jobId    = null;
let pollTimer = null;

// ── Model status polling ───────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const { ready, error } = await fetch('/api/status').then(r => r.json());
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-text');
    if (ready) {
      dot.className = 'w-2 h-2 rounded-full bg-green-400';
      txt.textContent = 'Model ready';
      txt.className = 'text-green-400';
    } else if (error) {
      dot.className = 'w-2 h-2 rounded-full bg-red-500';
      txt.textContent = 'Error: ' + error;
      txt.className = 'text-red-400';
    } else {
      setTimeout(checkStatus, 2500);
    }
  } catch {
    setTimeout(checkStatus, 3000);
  }
}
checkStatus();

// ── File selection ────────────────────────────────────────────────────────────
function handleDrop(e) {
  e.preventDefault();
  addFiles(Array.from(e.dataTransfer.files).filter(isCuda));
}
function handleFiles(list) {
  addFiles(Array.from(list).filter(isCuda));
}
function isCuda(f) { return f.name.endsWith('.cu') || f.name.endsWith('.cuh'); }
function addFiles(files) {
  for (const f of files)
    if (!selected.find(x => (x.webkitRelativePath || x.name) === (f.webkitRelativePath || f.name)))
      selected.push(f);
  renderFileList();
}
function clearFiles()    { selected = []; renderFileList(); }
function removeFile(i)   { selected.splice(i, 1); renderFileList(); }

function renderFileList() {
  const listEl  = document.getElementById('file-list');
  const itemsEl = document.getElementById('file-items');
  const countEl = document.getElementById('file-count');
  if (!selected.length) { listEl.classList.add('hidden'); return; }
  listEl.classList.remove('hidden');
  countEl.textContent = selected.length;
  itemsEl.innerHTML = selected.map((f, i) => `
    <div class="flex items-center justify-between bg-gray-900 border border-gray-800 rounded-lg px-3 py-1.5 text-sm">
      <span class="text-gray-300 truncate mr-3">${f.webkitRelativePath || f.name}</span>
      <div class="flex items-center gap-3 shrink-0">
        <span class="text-gray-600 text-xs">${(f.size/1024).toFixed(1)} KB</span>
        <button onclick="removeFile(${i})" class="text-gray-700 hover:text-red-400 transition-colors">✕</button>
      </div>
    </div>`).join('');
}

// ── Transpile ─────────────────────────────────────────────────────────────────
async function run() {
  if (!selected.length) return;
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  document.getElementById('upload-section').classList.add('hidden');
  document.getElementById('progress-section').classList.remove('hidden');

  const form = new FormData();
  for (const f of selected) form.append('files', f, f.webkitRelativePath || f.name);

  try {
    const res = await fetch('/api/jobs', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      alert('Error: ' + err.detail);
      reset(); return;
    }
    jobId = (await res.json()).job_id;
    pollTimer = setInterval(poll, 1500);
  } catch (e) {
    alert('Network error: ' + e.message);
    reset();
  }
}

async function poll() {
  const job = await fetch(`/api/jobs/${jobId}`).then(r => r.json());
  const pct = job.total ? (job.done / job.total) * 100 : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-fraction').textContent = `${job.done} / ${job.total}`;
  document.getElementById('progress-file').textContent = job.current ? 'Processing: ' + job.current : '';
  if (job.status === 'done') { clearInterval(pollTimer); showResults(job.results); }
}

// ── Results ───────────────────────────────────────────────────────────────────
function showResults(results) {
  document.getElementById('progress-section').classList.add('hidden');
  document.getElementById('results-section').classList.remove('hidden');
  const list = document.getElementById('results-list');
  list.innerHTML = '';
  for (const [name, r] of Object.entries(results)) {
    const el = document.createElement('div');
    el.className = 'bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden';
    const outName = name.replace('.cu', '.hip').replace('.cuh', '_hip.h');
    if (r.ok) {
      el.innerHTML = `
        <div class="flex items-center justify-between px-5 py-3 border-b border-gray-800">
          <div class="flex items-center gap-2 text-sm min-w-0">
            <span class="text-gray-400 truncate">${esc(name)}</span>
            <span class="text-gray-600 shrink-0">→</span>
            <span class="text-blue-400 truncate">${esc(outName)}</span>
          </div>
          <span class="ml-3 shrink-0 text-xs bg-green-900/50 text-green-300 border border-green-800/60 px-2 py-0.5 rounded-full">✓ Success</span>
        </div>
        <div class="grid grid-cols-2 divide-x divide-gray-800">
          <div>
            <div class="px-4 py-2 text-xs text-gray-500 border-b border-gray-800 uppercase tracking-wider font-medium">CUDA · Input</div>
            <pre class="overflow-auto max-h-80 text-xs p-4"><code class="language-cpp">${esc(r.cuda)}</code></pre>
          </div>
          <div>
            <div class="px-4 py-2 text-xs text-gray-500 border-b border-gray-800 uppercase tracking-wider font-medium">HIP · Output</div>
            <pre class="overflow-auto max-h-80 text-xs p-4"><code class="language-cpp">${esc(r.hip)}</code></pre>
          </div>
        </div>`;
    } else {
      el.innerHTML = `
        <div class="flex items-center justify-between px-5 py-3 border-b border-gray-800">
          <span class="text-sm text-gray-400">${esc(name)}</span>
          <span class="text-xs bg-red-900/50 text-red-300 border border-red-800/60 px-2 py-0.5 rounded-full">✗ Error</span>
        </div>
        <div class="px-5 py-4 text-sm text-red-400 font-mono">${esc(r.error)}</div>`;
    }
    list.appendChild(el);
    el.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));
  }
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function downloadZip() { if (jobId) window.location.href = `/api/jobs/${jobId}/download`; }
function reset() {
  clearInterval(pollTimer);
  selected = []; jobId = null;
  document.getElementById('run-btn').disabled = false;
  document.getElementById('upload-section').classList.remove('hidden');
  document.getElementById('progress-section').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');
  renderFileList();
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
