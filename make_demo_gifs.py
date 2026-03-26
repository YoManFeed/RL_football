"""Train PPO and produce GIF demos at different training stages.

Produces:
  gifs/stage_random.gif     — untrained (random) agent
  gifs/stage_50k.gif        — after  50 000 env steps
  gifs/stage_150k.gif       — after 150 000 env steps
  gifs/stage_300k.gif       — after 300 000 env steps
  gifs/learning_curve.png   — episode return vs env steps
"""

from __future__ import annotations

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))

from football_rl.wrappers.gym_env import FootballGymEnv
from football_rl.configs.defaults import make_default_config
from football_rl.training.ppo import PPOAgent, PPOConfig

# ── training config ───────────────────────────────────────────────────────────

SCENARIO      = "scenario_2_moving_wall"
TOTAL_STEPS   = 300_000
ROLLOUT_STEPS = 2048
GIF_FPS       = 18
GIF_W, GIF_H  = 480, 320
GIF_DIR       = Path("gifs")
CKPT_DIR      = Path("checkpoints")

# (milestone_step → label)
CAPTURE_AT = {
    0:       "random",
    50_000:  "50k",
    150_000: "150k",
    300_000: "300k",
}


# ── reward-shaped config ──────────────────────────────────────────────────────

def make_training_config():
    """Custom config with amplified dense rewards so PPO gets a useful signal."""
    cfg = make_default_config()
    cfg.rewards.ball_progress_scale = 2.0   # default 0.04 → ×50
    cfg.rewards.touch_reward = 0.5          # default 0.05 → ×10
    cfg.ball.allow_dribble = False          # kick-only
    # fix attack direction for stage-1: removes canonical/action mismatch (×3 speedup)
    cfg.randomization.randomize_attack_direction = False
    return cfg


# ── ball-approach shaping wrapper ─────────────────────────────────────────────

class BallApproachWrapper:
    """Potential-based shaping: reward getting closer to the ball.

    obs layout (flattened, no teammates/opponents):
      [0:11]  self block
      [11:17] ball block — [15:17] = normalize_position(ball-self + [60,40])
        => actual_rel_x = obs[15] * 60,  actual_rel_y = obs[16] * 40
    """

    APPROACH_SCALE = 3.0   # reward per unit of progress toward ball (was 0.4)

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._prev_dist: float | None = None

    @staticmethod
    def _dist_to_ball(obs: np.ndarray) -> float:
        rel_x = float(obs[15]) * 60.0
        rel_y = float(obs[16]) * 40.0
        return float((rel_x ** 2 + rel_y ** 2) ** 0.5)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_dist = self._dist_to_ball(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_dist = self._dist_to_ball(obs)
        if self._prev_dist is not None:
            # positive when approaching, negative when moving away
            shaping = self.APPROACH_SCALE * (self._prev_dist - curr_dist) / 120.0
            reward += shaping
        self._prev_dist = curr_dist if not (terminated or truncated) else None
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


# ── env factories ─────────────────────────────────────────────────────────────

def make_train_env():
    base = FootballGymEnv(
        SCENARIO,
        config=make_training_config(),
        flatten_observation=True,
        canonical_observation=True,
    )
    return BallApproachWrapper(base)


def make_render_env():
    cfg = make_default_config()
    cfg.ball.allow_dribble = False
    cfg.randomization.randomize_attack_direction = False
    return FootballGymEnv(
        SCENARIO,
        config=cfg,
        flatten_observation=True,
        canonical_observation=True,
        render_mode="rgb_array",
    )


# ── GIF helpers ───────────────────────────────────────────────────────────────

def _add_text(frame: np.ndarray, label: str, ret: float | None = None) -> np.ndarray:
    """Burn label + return value into the frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    text = label
    if ret is not None:
        text += f"  |  return {ret:.2f}"
    draw.rectangle([0, 0, img.width, 22], fill=(20, 20, 20))
    draw.text((6, 3), text, fill=(220, 220, 50))
    return np.array(img)


def render_episode(agent: PPOAgent, label: str, seed: int = 0) -> tuple[list[np.ndarray], float]:
    """Run the best of 5 episodes; returns (frames, episode_return)."""
    best_frames: list[np.ndarray] = []
    best_ret = -1e9

    for trial_seed in range(seed, seed + 5):
        env = make_render_env()
        obs, _ = env.reset(seed=trial_seed)
        frames = []
        ep_ret = 0.0
        done = False
        while not done:
            raw = env.render()
            if raw is not None:
                img = Image.fromarray(raw).resize((GIF_W, GIF_H), Image.BILINEAR)
                frames.append(np.array(img))
            action, _, _ = agent.select_action(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        env.close()
        if ep_ret > best_ret:
            best_ret = ep_ret
            best_frames = frames

    stamped = [_add_text(f, label, best_ret) for f in best_frames]
    return stamped, best_ret


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    import imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=GIF_FPS, loop=0)
    print(f"  saved  {path.name}  ({len(frames)} frames, {path.stat().st_size//1024} KB)")


# ── learning curve ────────────────────────────────────────────────────────────

def save_learning_curve(steps_log, return_log, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(return_log, dtype=float)
    w = min(30, max(1, len(arr)))
    smooth = np.convolve(arr, np.ones(w) / w, mode="valid")
    sx = steps_log[w - 1:]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps_log, return_log, alpha=0.2, color="steelblue", linewidth=0.8, label="episode return")
    ax.plot(sx, smooth, color="steelblue", linewidth=2.0, label=f"rolling mean ({w} ep)")

    colours = ["#e63946", "#2a9d8f", "#f4a261", "#264653"]
    for (step, lbl), col in zip(
        [(s, l) for s, l in sorted(CAPTURE_AT.items()) if s > 0], colours
    ):
        ax.axvline(step, color=col, linestyle="--", linewidth=1.2, alpha=0.9, label=f"GIF: {lbl}")

    ax.set_xlabel("Env steps", fontsize=11)
    ax.set_ylabel("Episode return", fontsize=11)
    ax.set_title(f"PPO on {SCENARIO}  (shaped rewards)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=140)
    plt.close(fig)
    print(f"  saved  {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    GIF_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    env = make_train_env()
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Scenario   : {SCENARIO}")
    print(f"obs_dim    : {obs_dim}   action_dim: {action_dim}")
    print(f"Total steps: {TOTAL_STEPS:,}")

    cfg = PPOConfig(
        rollout_steps  = ROLLOUT_STEPS,
        num_epochs     = 4,
        minibatch_size = 512,
        lr             = 3e-4,
        clip_epsilon   = 0.2,
        entropy_coef   = 0.01,
    )
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=cfg)

    # capture random-agent GIF
    print("\n[capture] random (untrained)")
    frames, ret = render_episode(agent, "Untrained  (random policy)", seed=0)
    save_gif(frames, GIF_DIR / "stage_random.gif")

    # training loop
    num_updates = TOTAL_STEPS // ROLLOUT_STEPS
    total_steps = 0
    steps_log: list[int]   = []
    return_log: list[float] = []
    captured: set[int] = {0}

    print(f"\nTraining PPO: {num_updates} updates × {ROLLOUT_STEPS} steps …\n")

    for update in range(1, num_updates + 1):
        last_val, rollout_stats = agent.collect_rollout(env)
        update_stats = agent.update(last_val)
        total_steps += ROLLOUT_STEPS

        if rollout_stats["num_episodes"] > 0:
            steps_log.append(total_steps)
            return_log.append(rollout_stats["mean_ep_return"])

        if update % 10 == 0:
            recent = np.mean(return_log[-30:]) if return_log else float("nan")
            print(
                f"  upd {update:4d}/{num_updates}  "
                f"steps {total_steps:>8,}  "
                f"ret(30ep) {recent:7.3f}  "
                f"ent {update_stats['entropy']:.2f}  "
                f"clip {update_stats['clip_frac']:.2f}"
            )

        # milestone GIF captures
        for milestone, label in sorted(CAPTURE_AT.items()):
            if milestone in captured or total_steps < milestone:
                continue
            captured.add(milestone)
            ckpt = CKPT_DIR / f"ppo_{SCENARIO}_{label}.pt"
            agent.save(ckpt)
            print(f"\n[capture] {label} ({total_steps:,} steps)")
            frames, ep_ret = render_episode(agent, f"PPO  {label} steps", seed=0)
            save_gif(frames, GIF_DIR / f"stage_{label}.gif")
            print()

    env.close()
    agent.save(CKPT_DIR / f"ppo_{SCENARIO}_final.pt")

    print("[plot] learning curve")
    save_learning_curve(steps_log, return_log, GIF_DIR / "learning_curve.png")

    print("\nAll done! Files in", GIF_DIR.resolve())
    for f in sorted(GIF_DIR.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:<42}  {size_kb:>5} KB")


if __name__ == "__main__":
    main()
