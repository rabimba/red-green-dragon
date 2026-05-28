# Red-Green-Dragon

CUDA ↔ AMD HIP transpilation tooling: Parquet/Arrow-indexed source pipelines, rule-based HIPify conversion, and local LLM inference.

## Prerequisites

- **Docker** with NVIDIA Container Toolkit (GPU passthrough)
- **NVIDIA GPU** for `stackv2_scripts/` (`nvcc` / `cuobjdump`)
- **Local Parquet or Arrow index files** for pipeline scripts (paths via CLI; scripts do not fetch remote indexes for you)
- **Inference**: Python 3 with `torch` and `transformers`; model weights under `models/` (see below)

The Docker image includes CUDA 12.8, LLVM/clang 18, `hipify-clang`, and Python packages `tqdm` and `datasets`. It does not install PyTorch or Transformers.

## Quick start (Docker)

Build and start an interactive shell with the repo mounted at `/workspace`:

```bash
docker build -t transpiler .
docker compose run --rm transpiler
```

Run a one-off command inside the container:

```bash
docker compose run --rm transpiler -c 'nvcc --version'
```

Optional host paths for large local trees are documented as commented volumes in `docker-compose.yml` (`DATASETS_HOST`, `BENCH_HOST`, etc.). Those mounts are not configured by default.

## What is not in this repository

| Item | Notes |
|------|--------|
| Model weights | Gitignored; place checkpoints under `models/` (see LLM inference) |
| Full Parquet/Arrow trees or benchmark trees | Provide your own paths; optional compose volume mounts only |
| Training or evaluation scripts | Not present in this repo |

## Directory layout

| Path | Purpose |
|------|---------|
| `stackv2_scripts/` | Download CUDA blobs, restore repo layout, clone repos, disassemble to SASS, pair assembly with source |
| `hipify/` | Batch `hipify-clang` conversion (flat dir or repo tree) |
| `infer_test.py` | Minimal script: load local source model, transpile embedded CUDA sample |
| `webapp/` | FastAPI UI for uploading `.cu` files and downloading HIP results |
| `models/` | Local weight directory (gitignored checkpoints) |
| `Dockerfile`, `docker-compose.yml` | Development environment |
| `assets/` | Project images (not required to run scripts) |

## `stackv2_scripts/` pipeline

Run inside the Docker container from `/workspace/stackv2_scripts`. Steps are ordered; each script takes local file paths via CLI flags.

1. **Download CUDA sources from a Parquet index**

   ```bash
   python3 write_dataset.py --save-dir /path/to/out --dataset-path /path/to/stack.parquet --num-threads 8
   ```

2. **Restore original repository paths**

   ```bash
   python3 create_repo_structure.py --source /path/to/flat --destination /path/to/structured --dataset-path /path/to/stack.parquet
   ```

3. **Clone top-N repos by CUDA file count**

   ```bash
   python3 clone_repos.py --destination /path/to/structured --arrow-path /path/to/stack.arrow --first-n 100
   ```

4. **Compile CUDA and extract SASS** (requires NVIDIA GPU)

   ```bash
   python3 disassemble_cuda.py --dataset /path/to/structured --sass-dir /path/to/sass --arch sm_80 --arrow-path /path/to/stack.arrow
   ```

   Uses `nvcc -std=c++17 -Xcompiler=-Os -DNDEBUG -w -arch=<arch>` and `cuobjdump --dump-sass`.

5. **Pair assembly outputs with source files**

   ```bash
   python3 grab_assembly_source_from_structured.py --as-dir /path/to/sass --structured-dir /path/to/structured --sources-dir /path/to/out --arrow-path /path/to/stack.arrow
   ```

## Rule-based CUDA → HIP (`hipify/`)

`hipify-clang` is installed in the Docker image. Example single-file conversion (from `hipify/run.sh`):

```bash
cd hipify
hipify-clang --clang-resource-directory=/usr/lib/llvm-18/lib/clang/18 \
  --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_80 -o sample.hip sample.cu
```

**Flat directory of `.cu` files:**

```bash
python3 hip_from_folder.py --source /path/to/cuda --destination /path/to/hip --threads 8
```

**Preserve repo tree** (requires Arrow metadata):

```bash
python3 hip_from_repo_structure.py --source /path/to/structured --destination /path/to/hip --arrow-path /path/to/stack.arrow --threads 8
```

## LLM inference

Place weights on disk before running inference. Default scripts load **`./models/red-green-dragon-src-7b`** with `device_map="auto"`.

| Size | Local path |
|------|------------|
| 1.5B | `models/red-green-dragon-src-1.5b/` |
| 3B   | `models/red-green-dragon-src-3b/` |
| 7B   | `models/red-green-dragon-src-7b/` |

Assembly-level checkpoints (SASS variants) can use matching names under `models/`, for example `models/red-green-dragon-smA100-1.5b/`. Copy or download full checkpoint trees (config, tokenizer, `*.safetensors`) into those directories.

**CLI test:**

```bash
pip install torch transformers
python3 infer_test.py
```

**Web UI:**

```bash
pip install fastapi uvicorn python-multipart torch transformers
python3 webapp/app.py
```

Open `http://0.0.0.0:7860`. The app transpiles uploaded CUDA sources using the same prompt template as `infer_test.py`.

### Multi-GPU

With `device_map="auto"`, Transformers spreads layers across visible GPUs. Limit devices, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python3 infer_test.py
```
