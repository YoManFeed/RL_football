# football-rl

`football-rl` is a deterministic 2D football simulator for reinforcement learning. The project is structured as a platform rather than a single scenario prototype: the simulator core is independent from Gymnasium and multi-agent wrappers, rewards are modular, scenarios are registered through a factory, and rendering is optional.

## Features

- Python 3.11+
- `numpy`-based deterministic 2D physics
- Gymnasium-compatible single-agent wrapper
- parallel-style multi-agent wrapper for centralized-critic / decentralized-execution workflows
- headless and pygame render modes
- domain randomization at `reset()`
- canonical observations for left/right symmetry
- modular rewards and scripted baselines
- curriculum and self-play utilities
- smoke tests for reset, step, goal, kick, scenario loading, and multi-agent mode

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

Run scenario 1 with rendering:

```bash
python examples/render_scenario1.py
```

Headless rollout:

```bash
python examples/headless_rollout.py
```

Parallel multi-agent scenario 7:

```bash
python examples/multiagent_scenario7.py
```

Run tests:

```bash
pytest -q
```

## Creating an environment by scenario name

Single-agent:

```python
from football_rl.wrappers.gym_env import FootballGymEnv

env = FootballGymEnv(
    scenario_name="scenario_1_single_striker",
    render_mode=None,
    canonical_observation=True,
    flatten_observation=False,
)
obs, info = env.reset(seed=7)
```

Multi-agent:

```python
from football_rl.wrappers.multi_agent_env import ParallelFootballEnv

env = ParallelFootballEnv(
    scenario_name="scenario_7_full_match",
    render_mode=None,
    canonical_observation=True,
)
obs, infos = env.reset(seed=7)
```

## Why randomization and symmetry matter

Without randomization the agent quickly overfits to geometry shortcuts. If the ball always spawns at the same point, if the team always attacks to the right, or if the wall always moves with the same phase, the policy can memorize a narrow opening sequence instead of learning football-like behavior.

This project randomizes side of attack, team-color mapping, initial placements, ball spawn offsets, obstacle sizes and speeds, goalkeeper start position, and episode length within scenario-safe bounds. Observations also expose `attack_direction`, `team_id`, and explicit own/opponent goal positions.

The `canonical_observation` option mirrors the state so that “attack forward” always corresponds to positive X from the agent’s perspective. This makes transfer across left/right sides significantly easier.

## Curriculum overview

The curriculum is organized as increasing-complexity stages:

- **Stage A**: movement and ball contact
- **Stage B**: strike into empty goal with reset randomization
- **Stage C**: timing against a moving wall
- **Stage D**: pass-then-goal
- **Stage E**: goalkeeper against previously trained attackers
- **Stage F**: self-play fine-tuning in team-vs-team mode

See `examples/staged_training_stub.py` and `football_rl/training/curriculum.py`.

## Transfer learning and role specialization

A field player and a goalkeeper are not treated as identical. The training utilities support two patterns:

1. shared encoder + role-specific policy heads
2. separate role-specific policies with encoder weight transfer

The sample `SharedEncoderRolePolicy` in `football_rl/training/policies.py` supports:

- loading a checkpoint from a previous stage
- freezing the shared encoder
- replacing a role head when the action distribution changes
- continuing training rather than inference-only reuse

## Self-play

The self-play utilities in `football_rl/policies/self_play.py` support:

- frozen opponents
- a pool of historical checkpoints
- sampling from multiple old opponents to avoid overfitting to a single version

## Project layout

```text
football_rl_project/
  football_rl/
    configs/
    core/
    entities/
    physics/
    policies/
    render/
    rewards/
    scenarios/
    training/
    utils/
    wrappers/
  examples/
  tests/
  README.md
  pyproject.toml
```
