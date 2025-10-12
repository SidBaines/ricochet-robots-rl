#%%
"""
Bank Curriculum Preview

This Jupyter-style script mirrors the bank-based curriculum setup from `train_agent.py`.
It loads curriculum levels from `configs/curriculum_config_default.json`, uses the `puzzle_bank`,
and renders a few RGB examples from each level.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

import json
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from src.env.curriculum import create_bank_curriculum_manager, BankCurriculumWrapper
from src.env.puzzle_bank import PuzzleBank, SpecKey


#%%
# Configuration
BANK_DIR = str((ARTIFACTS_ROOT / "puzzle_bank").resolve())
CONFIG_PATH = str((PROJECT_ROOT / "configs" / "curriculum_config_default.json").resolve())
EXAMPLES_PER_LEVEL = 5
OBS_MODE = "rgb_image"
CHANNELS_FIRST = True

# Curriculum thresholds (aligned with train_agent.py defaults)
SUCCESS_RATE_THRESHOLD = 0.8
MIN_EPISODES_PER_LEVEL = 100
SUCCESS_RATE_WINDOW_SIZE = 200
ADVANCEMENT_CHECK_FREQUENCY = 50
VERBOSE = True


#%%
# Load curriculum levels from JSON and convert spec_keys to SpecKey
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_data: Dict[str, Any] = json.load(f)

levels: List[Dict[str, Any]] = config_data.get("levels", [])
for level in levels:
    spec_dict = level["spec_key"]
    if isinstance(spec_dict, dict):
        level["spec_key"] = SpecKey(
            height=spec_dict["height"],
            width=spec_dict["width"],
            num_robots=spec_dict["num_robots"],
            edge_t_per_quadrant=spec_dict["edge_t_per_quadrant"],
            central_l_per_quadrant=spec_dict["central_l_per_quadrant"],
            generator_rules_version=spec_dict.get("generator_rules_version", 1),
        )

print(f"Loaded {len(levels)} curriculum levels from {CONFIG_PATH}")


#%%
# Initialize puzzle bank and curriculum manager
bank = PuzzleBank(BANK_DIR)
bank_manager = create_bank_curriculum_manager(
    bank=bank,
    curriculum_levels=levels,
    success_rate_threshold=SUCCESS_RATE_THRESHOLD,
    min_episodes_per_level=MIN_EPISODES_PER_LEVEL,
    success_rate_window_size=SUCCESS_RATE_WINDOW_SIZE,
    advancement_check_frequency=ADVANCEMENT_CHECK_FREQUENCY,
    verbose=VERBOSE,
)


#%%
# Utility to create a wrapper bound to the manager's current level

def make_wrapper_for_current_level(render_mode: str = "rgb") -> BankCurriculumWrapper:
    return BankCurriculumWrapper(
        bank=bank,
        curriculum_manager=bank_manager,
        obs_mode=OBS_MODE,
        channels_first=CHANNELS_FIRST,
        render_mode=render_mode,
        verbose=VERBOSE,
    )


#%%
# Render a few examples per level in RGB mode
plt.rcParams["figure.figsize"] = (4 * EXAMPLES_PER_LEVEL, 4)

for level_idx, level_spec in enumerate(levels):
    bank_manager.current_level = level_idx
    env = make_wrapper_for_current_level(render_mode="rgb")
    fig, axes = plt.subplots(1, EXAMPLES_PER_LEVEL)
    if EXAMPLES_PER_LEVEL == 1:
        axes = [axes]
    for i in range(EXAMPLES_PER_LEVEL):
        obs, info = env.reset()
        rgb = env.render()
        if isinstance(rgb, np.ndarray):
            axes[i].imshow(rgb)
            axes[i].axis("off")
            title = f"L{level_idx}: {level_spec['name']}\nopt_len={info.get('optimal_length', '?')} robots_moved={info.get('robots_moved', '?')}"
            axes[i].set_title(title, fontsize=9)
        else:
            axes[i].text(0.5, 0.5, "No RGB available", ha="center", va="center")
            axes[i].axis("off")
    fig.suptitle(f"Curriculum Level {level_idx}: {level_spec['name']}")
    plt.tight_layout()
    plt.show()
    env.close()

print("Preview complete.")

# %%
