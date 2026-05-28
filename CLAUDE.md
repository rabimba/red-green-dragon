# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Red-Green-Dragon** provides CUDA ↔ AMD HIP transpilation tooling: source-level and assembly-level translation pipelines, rule-based HIPify, and local LLM inference.

- Models: Local weights in `models/` directory (1.5B, 3B, 7B variants for source; 1.5B, 3B for assembly)

## Development Environment

All development should be done inside the Docker container, which provides CUDA 12.8, LLVM 18, and all required GPU tooling.

```bash
# Build the container
docker build -t transpiler .

# Run with GPU access (as configured in docker-compose.yml)
docker compose run transpiler
```

`docker-compose.yml` mounts the repo at `/workspace`; optional commented volume lines can point at host data directories. Scripts expect NVIDIA GPU access for CUDA compilation and AMD GPU for HIP conversion.

## Repository Structure

There is no traditional build system or test suite — this is a GPU tooling and data pipeline collection.

### `stackv2_scripts/` — CUDA source pipeline

Scripts are meant to be run in order:

1. `write_dataset.py` — Downloads CUDA files from a local parquet index
2. `create_repo_structure.py` — Reconstructs original repo folder structures from flat downloaded files
3. `clone_repos.py` — Clones the top N repos (by CUDA file count) for richer context
4. `disassemble_cuda.py` — Compiles CUDA files with `nvcc` and extracts SASS assembly via `cuobjdump`
5. `grab_assembly_source_from_structured.py` — Pairs SASS assembly with its original source files

Key implementation patterns in these scripts:
- `ThreadPoolExecutor` with 8–32 workers for parallel file processing
- `subprocess` calls to `nvcc` and `cuobjdump` with 60–90 second timeouts
- `datasets` library with Arrow format for local data loading

### `hipify/` — CUDA-to-HIP Conversion Utilities

- `hip_from_folder.py` — Batch hipify a flat directory of `.cu` files
- `hip_from_repo_structure.py` — Hipify while preserving original repository structure
- `run.sh` — Example `hipify-clang` invocation with correct flags
- `sample.cu` / `sample.hip` — Reference example of CUDA → HIP conversion

## Key Tool Invocations

CUDA compilation (used in `disassemble_cuda.py`):
```bash
nvcc -std=c++17 -Xcompiler=-Os -arch=sm_80 -cubin <file.cu>
cuobjdump --dump-sass <file.cubin>
```

HIP conversion (used in hipify scripts and `run.sh`):
```bash
hipify-clang --clang-resource-directory /usr/lib/llvm-18/lib/clang/18 \
  --cuda-gpu-arch sm_80 <file.cu>
```

## Data Access

Scripts use the `datasets` and `boto3` / `smart_open` libraries to load local Arrow/Parquet files and stream from Software Heritage S3. Pipeline scripts accept local file paths via CLI arguments — no remote downloads required.
