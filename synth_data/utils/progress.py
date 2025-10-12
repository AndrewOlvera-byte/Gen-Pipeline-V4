from __future__ import annotations
from typing import Optional
from tqdm import tqdm


def create_progress(total_episodes: int, total_steps_hint: Optional[int] = None):

    ep_bar = tqdm(total=total_episodes, desc="Episodes", position=0, leave=True)
    step_bar = None
    if total_steps_hint is not None:
        step_bar = tqdm(total=total_steps_hint, desc="Steps", position=1, leave=True)
    return ep_bar, step_bar


def episode_progress(current_step: int, max_steps: Optional[int] = None) -> str:

    if max_steps is None or max_steps <= 0:
        return f"step {current_step}"
    return f"step {current_step}/{max_steps}"

