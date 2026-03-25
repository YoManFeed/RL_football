"""PPO training script for Football RL.

Usage:
    python train_ppo.py
    python train_ppo.py --scenario scenario_1_single_striker --total_steps 500000
    python train_ppo.py --load checkpoints/ppo_stage1.pt --scenario scenario_2_moving_wall
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure the package is importable when running from this directory
sys.path.insert(0, str(Path(__file__).parent))

from football_rl.wrappers.gym_env import FootballGymEnv
from football_rl.training.ppo import PPOAgent, PPOConfig
from football_rl.configs.defaults import make_default_config


def make_training_config():
    """Training config that forces kick-only behavior."""
    cfg = make_default_config()
    cfg.ball.allow_dribble = False          # kick-only, no dragging
    # fix attack direction for stage-1: removes canonical/action mismatch (×3 speedup)
    cfg.randomization.randomize_attack_direction = False
    return cfg


def save_learning_curve(steps_log: list[int], return_log: list[float], path: Path) -> None:
    """Plot and save learning curve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(return_log, dtype=float)
    w = min(30, max(1, len(arr)))
    smooth = np.convolve(arr, np.ones(w) / w, mode="valid")
    sx = steps_log[w - 1:]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps_log, return_log, alpha=0.2, color="steelblue", linewidth=0.8, label="episode return")
    ax.plot(sx, smooth, color="steelblue", linewidth=2.0, label=f"rolling mean ({w} ep)")

    ax.set_xlabel("Env steps", fontsize=11)
    ax.set_ylabel("Episode return", fontsize=11)
    ax.set_title("PPO Training Progress", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=140)
    plt.close(fig)
    print(f"  saved  {path.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="scenario_1_single_striker")
    p.add_argument("--total_steps", type=int, default=500_000)
    p.add_argument("--rollout_steps", type=int, default=2048)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.001)
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--minibatch_size", type=int, default=512)
    p.add_argument("--load", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--log_interval", type=int, default=10, help="Log every N updates")
    p.add_argument("--save_interval", type=int, default=50, help="Save checkpoint every N updates")
    p.add_argument("--device", default="cpu")
    p.add_argument("--plot_dir", default="plots", help="Directory to save learning curve plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(exist_ok=True)

    # Environment
    env = FootballGymEnv(
        scenario_name=args.scenario,
        config=make_training_config(),
        flatten_observation=True,
        canonical_observation=True,
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Scenario : {args.scenario}")
    print(f"Obs dim  : {obs_dim}")
    print(f"Action dim: {action_dim}")

    # Agent
    cfg = PPOConfig(
        rollout_steps   = args.rollout_steps,
        num_epochs      = args.num_epochs,
        minibatch_size  = args.minibatch_size,
        clip_epsilon    = args.clip,
        lr              = args.lr,
        entropy_coef    = args.entropy_coef,
    )
    agent = PPOAgent(
        obs_dim     = obs_dim,
        action_dim  = action_dim,
        hidden_dim  = args.hidden_dim,
        config      = cfg,
        device      = args.device,
    )

    if args.load:
        agent.load(args.load)
        print(f"Loaded checkpoint: {args.load}")

    # Training loop
    total_env_steps = 0
    update_idx = 0
    num_updates = args.total_steps // args.rollout_steps

    print(f"\nStarting PPO training: {num_updates} updates × {args.rollout_steps} steps "
          f"= {num_updates * args.rollout_steps:,} env steps\n")

    all_returns: list[float] = []
    steps_log: list[int] = []

    for _ in range(num_updates):
        last_value, rollout_stats = agent.collect_rollout(env)
        update_stats = agent.update(last_value)

        total_env_steps += args.rollout_steps
        update_idx += 1

        if rollout_stats["num_episodes"] > 0:
            all_returns.append(rollout_stats["mean_ep_return"])
            steps_log.append(total_env_steps)

        if update_idx % args.log_interval == 0:
            recent = np.mean(all_returns[-20:]) if all_returns else float("nan")
            print(
                f"Update {update_idx:4d}/{num_updates} | "
                f"steps {total_env_steps:>8,} | "
                f"return(20ep) {recent:7.3f} | "
                f"pg {update_stats['pg_loss']:6.4f} | "
                f"v {update_stats['v_loss']:6.4f} | "
                f"ent {update_stats['entropy']:5.3f} | "
                f"clip {update_stats['clip_frac']:.2f}"
            )

        if update_idx % args.save_interval == 0:
            ckpt_path = save_dir / f"ppo_{args.scenario}_step{total_env_steps}.pt"
            agent.save(ckpt_path)
            print(f"  → saved {ckpt_path}")

    # Final checkpoint
    final_path = save_dir / f"ppo_{args.scenario}_final.pt"
    agent.save(final_path)
    print(f"\nTraining done. Final checkpoint: {final_path}")
    
    # Save learning curve
    plot_path = plot_dir / f"learning_curve_{args.scenario}.png"
    save_learning_curve(steps_log, all_returns, plot_path)
    
    env.close()


if __name__ == "__main__":
    main()
