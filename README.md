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
- modular rewards with per-term breakdown and TensorBoard logging
- potential-based ball-approach reward shaping
- curriculum and self-play utilities
- smoke tests for reset, step, goal, kick, scenario loading, and multi-agent mode

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install torch  # required for PPO training
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

## Training

### Stage 1 — Single striker vs empty goal

```bash
python train_ppo.py \
    --scenario scenario_1_single_striker \
    --total_steps 500000 \
    --tensorboard \
    --ball_approach \
    --approach_scale 3.0 \
    --ball_progress_scale 3.0 \
    --ball_progress_euclidean \
    --goal_reward 1.0 \
    --touch_reward 0.01 \
    --no_dribble \
    --fix_attack_dir \
    --entropy_coef 0.005
```

### Stage 2 — Moving wall (transfer from stage 1)

```bash
python train_ppo.py \
    --scenario scenario_2_moving_wall \
    --total_steps 500000 \
    --load checkpoints/ppo_scenario_1_single_striker_final.pt \
    --tensorboard \
    --ball_approach \
    --approach_scale 3.0 \
    --ball_progress_scale 3.0 \
    --ball_progress_euclidean \
    --goal_reward 1.0 \
    --touch_reward 0.01 \
    --no_dribble \
    --fix_attack_dir \
    --entropy_coef 0.005
```

### Monitoring with TensorBoard

```bash
tensorboard --logdir runs
```

### Generate demo GIF from a trained checkpoint

```bash
python generate_gif.py \
    --checkpoint checkpoints/ppo_scenario_1_single_striker_final.pt \
    --scenario scenario_1_single_striker \
    --output gifs/demo.gif
```

## Creating an environment by scenario name

Single-agent:

```python
from football_rl.wrappers.gym_env import FootballGymEnv

env = FootballGymEnv(
    scenario_name=”scenario_1_single_striker”,
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
    scenario_name=”scenario_7_full_match”,
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

- **Stage A / Scenario 1**: single striker vs empty goal — learn to approach ball and score
- **Stage B / Scenario 2**: timing against a moving wall obstacle
- **Stage C / Scenario 3**: pass-then-goal — two players, pass required before goal counts
- **Stage D / Scenario 4**: pass + moving wall
- **Stage E / Scenario 5**: pass + scripted goalkeeper
- **Stage F / Scenario 6**: 1v1 duel
- **Stage G / Scenario 7**: full 3v3 match with self-play

See `examples/staged_training_stub.py` and `football_rl/training/curriculum.py`.

## Challenges faced and lessons learned

### 1. Sparse reward problem

The main reward signal (goal = 5.0) is large but rare. Without dense intermediate signals, the agent explores randomly for thousands of steps before accidentally scoring, making credit assignment nearly impossible.

**What didn’t work:**
- Small dense rewards alone (ball progress ~0.0003/step, touch 0.05) were orders of magnitude too small relative to the goal reward. The agent couldn’t distinguish signal from noise.

**What worked:**
- **Ball-approach potential shaping** (`BallApproachWrapper`, scale=3.0): rewards the agent for getting closer to the ball. This is the single most important addition — without it, the agent has no reason to move toward the ball at all.
- **Euclidean ball progress** (scale=3.0): rewards ball movement toward the opponent goal center (not just x-axis). Gives directional credit for kicks.
- These two dense signals together create a clear learning curriculum: approach ball → kick toward goal → score.

### 2. Reward hacking — possession farming

When `touch_reward` was set to 0.5 (to make ball contact more rewarding), the agent discovered it could farm `possession_gain` events by kicking the ball away and immediately re-touching it in a loop. At 1M steps it reached return=136 while scoring 0 goals — 1700+ possession events per rollout.

**Diagnosis:** TensorBoard reward breakdown showed `StealRewardTerm` dominating at 0.43/step while `GoalRewardTerm` dropped to zero. The per-term logging was essential to spotting this.

**Fix:** Set `touch_reward=0.01`. The ball-approach wrapper already provides dense reward for approaching the ball, making high touch reward redundant and exploitable.

### 3. Value function instability from large goal reward

With `goal_reward=5.0`, episodes where a goal happened had returns ~5x higher than non-goal episodes. The value function couldn’t model this bimodal distribution, leading to:
- `v_loss` of 10–26 (should be <1)
- `explained_variance` of 0.01–0.35 (should be >0.5)
- Near-zero `clip_fraction` — policy barely updating because advantages were noise

**Fix:** Reduced `goal_reward` from 5.0 to 1.0. This brought v_loss down to 0.007, explained_var up to 0.77, and the policy started learning kick direction.

### 4. Entropy collapse vs entropy stagnation

- With `entropy_coef=0.01` (default) and weak rewards, entropy *increased* — the policy got more random because there was no useful gradient signal.
- With `entropy_coef=0.001`, entropy decreased but too aggressively.
- `entropy_coef=0.005` with strong dense rewards gave steady entropy decrease (5.5 → 3.8) as the policy sharpened.

### 5. Kick-only mode and fixed attack direction

Two environment simplifications that dramatically accelerated early-stage learning:
- `allow_dribble=False`: ball never sticks to the player. Forces discrete kick behavior, which is easier to learn than continuous dribbling.
- `randomize_attack_direction=False`: always attack in the same direction. Removes the need to generalize across left/right before the basic skill is learned.

These can be relaxed in later curriculum stages.

### Training diagnostics added

To diagnose the above issues, we added comprehensive logging to the training pipeline:

- **Per-term reward breakdown**: `RewardManager` returns both total reward and per-term contributions. This was critical for spotting reward hacking.
- **Event statistics**: goals, kicks, passes, steals, wall bounces, out-of-bounds per rollout.
- **PPO diagnostics**: explained variance, approximate KL divergence, gradient norms, advantage statistics.
- **Policy statistics**: per-action-dimension mean and std from both the rollout data and the learned log_std parameter.
- **TensorBoard integration**: all metrics logged under `rollout/`, `losses/`, `diagnostics/`, `rewards/`, `events/`, `actions/`, `policy/` groups.

### Current training results

| Stage | Scenario | Goals/rollout | explained_var | Status |
|-------|----------|--------------|---------------|--------|
| 1 | Single striker | 44 | 0.77 | Solved |
| 2 | Moving wall | 62 | 0.93 | Solved |
| 3+ | Pass-then-goal | — | — | Not yet attempted |

### Recommended reward configuration

After multiple iterations, the following configuration reliably trains stages 1–2:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `goal_reward` | 1.0 | Low enough for stable value function |
| `ball_progress_scale` | 3.0 | Strong directional signal toward goal |
| `ball_progress_euclidean` | True | Credit for lateral movement, not just x-axis |
| `touch_reward` | 0.01 | Prevents possession farming exploit |
| `approach_scale` | 3.0 | Dense reward for moving toward ball |
| `entropy_coef` | 0.005 | Balanced exploration/exploitation |
| `allow_dribble` | False | Simplifies early learning |
| `fix_attack_dir` | True | Removes left/right generalization burden |

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
football_rl/
  configs/          # SimulatorConfig, RewardConfig, etc.
  core/             # SoccerSimulator, actions, observations, events
  entities/         # Player, Ball, Obstacle
  physics/          # Collision detection, movement
  policies/         # Scripted baselines, self-play
  render/           # Pygame renderer
  rewards/          # RewardManager, per-term breakdown, 8 reward terms
  scenarios/        # 7 scenarios (single striker → full match)
  training/         # PPO agent, curriculum, transfer learning
  utils/            # Gym compat, math, seeding
  wrappers/         # FootballGymEnv, ParallelFootballEnv, BallApproachWrapper
examples/           # Render, headless, multi-agent, training examples
tests/              # Pytest smoke tests
train_ppo.py        # Main training script with TensorBoard support
generate_gif.py     # GIF generation from checkpoints
make_demo_gifs.py   # Full training + GIF pipeline
```
