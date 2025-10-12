## Synthetic Data Generation for FetchReach-v3 (Minari)

This folder provides a CPU-only entrypoint to generate an expert BC dataset for `FetchReach-v3` using a heuristic expert controller and stores it as a Minari dataset under the repository `data/` directory.

### Install (local CPU)

```bash
pip install -r synth_data/requirements.txt
```

If you see Mujoco-related errors, ensure you have Mujoco runtime installed or use your Docker Compose setup which already includes it.

### Run

```bash
python synth_data/gen.py
```

Environment variables (optional):

- `ENV_ID` (default: `FetchReach-v3`)
- `DATASET_ID` (default: `fetchreach_v3_expert_bc_v1`)
- `EPISODES` (default: `400`)
- `SEED` (default: `0`)

The script prints per-episode progress and a TLDR quality summary at the end. The resulting dataset is saved under `data/<DATASET_ID>` at your repo root and can be loaded by the training pipeline via the Minari dataset loader.

### What gets saved

Episodes with arrays: `observations`, `actions`, `rewards`, `terminations`, `truncations` following Gymnasium v1 conventions. The dataset is compatible with `src/components/datasets/minari_loader.py`, which flattens episodes for BC.

### Notes

- The expert policy here is a simple PD-like controller that moves the gripper towards the desired goal (no GPU required). If you later replace it with a learned PPO policy, ensure it runs on CPU (`device='cpu'`) and keep the observation flattened to match the BC loader expectations.

