# General Pipeline V4

## Pain Points Removed

- Lightning module boiler plate - gone
- Over abstraction - gone
General BC -> RL pipeline, with space for data gen

## Run outputs and reports

- Hydra sets a unique working directory per run under `outputs/${exp.name}/${now}`.
- Within each run directory, a `reports/` folder now contains:
  - `config_full.yaml` (fully resolved config)
  - `overrides.txt` (Hydra CLI overrides used for the run)
  - `summary.json` (best metric, total time, checkpoint paths)
  - `metrics.jsonl` (JSONL log of training/eval metrics; default offline logger)
  - `graph.png` (plot of key metrics tracked by the tqdm progress bar)
  - `checkpoints/step_*.pt` (plain .pt files with step index)
  - `topk.json` (index of top-K checkpoints by objective)

## Containerized Dev Usage (Team Workflow)

This repo is designed to run fully in Docker with GPU acceleration and live-mounted source for rapid iteration.

### Prerequisites

- Docker with NVIDIA GPU support
  - Linux: `nvidia-container-toolkit` installed and configured
  - Windows 11: WSL2 + Docker Desktop + GPU support enabled
Logging defaults to offline JSONL under `reports/metrics.jsonl`. WandB integration has been removed to keep the pipeline minimal.

### Build once, run many

The Compose file in `docker/docker-compose.yaml` builds an image using `docker/Dockerfile`.

- Build (or rebuild after dependency changes):
  - Linux/macOS:
    ```bash
    BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime docker compose -f docker/docker-compose.yaml build --no-cache
    ```
  - Windows PowerShell:
    ```powershell
    $env:BASE_IMAGE="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"; docker compose -f docker/docker-compose.yaml build --no-cache
    ```

Notes:
- You can omit `BASE_IMAGE` to use the default from the Dockerfile.
- Source and `outputs/` are volume-mounted, so code edits are reflected immediately without rebuilds.

### Runtime entrypoint and services

- The image default command is `python scripts/train.py` (see `docker/Dockerfile`). Compose services override or reuse this:
  - `trainer` service runs `python scripts/train.py` (training entrypoint)
  - `search` profile runs `python scripts/search.py -m +hpo=optuna +hpo.space=ppo_basic` (example; adjustable)
  - `eval` profile runs `python scripts/eval.py` (evaluates and writes `reports/summary.json`)

### Running training in the container

By default, the `trainer` service runs `scripts/train.py` using Hydra configs in `config/`.

- Start a standard training run (uses defaults from `config/defaults.yaml`):
  ```bash
  docker compose -f docker/docker-compose.yaml up trainer
  ```

- Override experiment/configs via Hydra CLI (examples):
  ```bash
  # Example: run BC on FetchReach
  docker compose -f docker/docker-compose.yaml run --rm trainer \
    python scripts/train.py +exp=fetch_bc +env=fetch_reach +dataset=minari \
    +trainer=bc +model=mlp_gauss +loss=bc_gaussian_nll

  # Example: run PPO on FetchReach
  docker compose -f docker/docker-compose.yaml run --rm trainer \
    python scripts/train.py +exp=fetch_ppo +env=fetch_reach +trainer=ppo \
    +model=td_actor_critic +loss=ppo

  # Example: quick debug presets (small steps/rollouts)
  docker compose -f docker/docker-compose.yaml run --rm trainer \
    python scripts/train.py +exp=quick_debug_bc
  ```

Artifacts:
- Each run writes under `outputs/${exp.name}/${now}/` with reports in `reports/`.

### Quick debug runs (Windows and Linux)

- Linux/macOS example (offline logging):
  ```bash
  WANDB_MODE=offline docker compose -f docker/docker-compose.yaml run --rm trainer \
    python scripts/train.py +exp=quick_debug_bc
  ```

- Windows PowerShell example (offline logging):
  ```powershell
  $env:WANDB_MODE="offline"; docker compose -f docker/docker-compose.yaml run --rm trainer `
    python scripts/train.py +exp=quick_debug_bc
  ```

You can swap to PPO quick debug similarly:
```bash
docker compose -f docker/docker-compose.yaml run --rm trainer \
  python scripts/train.py +exp=quick_debug_ppo
```

### Using the Optuna search (multirun)

This repo includes Hydra's Optuna Sweeper integration. Use the `-m` multirun flag, select the Optuna config, and pick a parameter space file from `config/hpo/`.

- Basic CLI usage from host or inside container shell:
  ```bash
  # BC search with a basic search space
  python scripts/search.py -m +hpo=optuna +hpo.space=space_bc_basic

  # PPO search with a basic search space
  python scripts/search.py -m +hpo=optuna +hpo.space=space_ppo_basic
  ```

- Using docker compose directly:
  ```bash
  docker compose -f docker/docker-compose.yaml run --rm trainer \
    python scripts/search.py -m +hpo=optuna +hpo.space=space_ppo_basic
  ```

- Useful overrides:
  ```bash
  # set number of trials and jobs
  python scripts/search.py -m +hpo=optuna +hpo.space=space_ppo_basic \
    hydra.sweeper.n_trials=50 hydra.sweeper.n_jobs=1

  # choose objective key exposed by evaluator
  python scripts/search.py -m +hpo=optuna +hpo.space=space_bc_basic \
    hpo.objective_key=eval/return_mean

  # persist Optuna study to sqlite file under outputs
  python scripts/search.py -m +hpo=optuna +hpo.space=space_bc_basic \
    hydra.sweeper.storage=sqlite:///outputs/optuna.db
  ```

- Outputs: each trial runs under its own Hydra run dir with `reports/summary.json`. A `topk.json` index is kept at the sweep root (see `scripts/search.py`).

### Interactive dev shell inside the container

Get a GPU-enabled shell with the project mounted at `/app`:
```bash
docker compose -f docker/docker-compose.yaml run --rm trainer bash
```
Examples from the shell:
```bash
python scripts/train.py +exp=fetch_ppo
pytest -q  # if you add tests
python -c "import torch, torch.version; print(torch.cuda.is_available(), torch.version.cuda)"
```

### Environment variables

The Compose file forwards:
- `MUJOCO_GL`: defaults to `egl`
- `NVIDIA_VISIBLE_DEVICES`: `all` (default) or specific GPU index(es)
- `NVIDIA_DRIVER_CAPABILITIES`: `all` (default)

### Data and volumes

- Dataset paths under the repo (e.g., `data/…`) are available inside the container at `/app/data`.
- Outputs are persisted to host `outputs/` via a bind mount.

### GPU troubleshooting

- Verify GPU inside container:
  ```bash
  docker compose -f docker/docker-compose.yaml run --rm trainer bash -lc "nvidia-smi || true; python -c 'import torch; print(torch.cuda.is_available())'"
  ```
- If `torch.cuda.is_available()` is `False`:
  - Linux: ensure `nvidia-container-toolkit` is installed and `--gpus all` is supported by Docker; restart Docker.
  - Windows: ensure WSL2 backend is enabled, GPU support on, and your driver is up to date. Try restarting Docker Desktop.

### Customizing the base image

You can change the PyTorch CUDA runtime used for the image:
```bash
BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime \
  docker compose -f docker/docker-compose.yaml build
```

The Dockerfile installs Python deps from `requirements.txt` first for better cache reuse.