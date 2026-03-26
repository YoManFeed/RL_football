"""PPO training script for Football RL.

Usage:
    python train_ppo.py
    python train_ppo.py --scenario scenario_1_single_striker --total_steps 500000
    python train_ppo.py --load checkpoints/ppo_stage1.pt --scenario scenario_2_moving_wall
    python train_ppo.py --tensorboard  # enable TensorBoard logging
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make sure the package is importable when running from this directory
sys.path.insert(0, str(Path(__file__).parent))

from football_rl.configs.defaults import SimulatorConfig, make_default_config
from football_rl.wrappers.gym_env import FootballGymEnv
from football_rl.training.ppo import PPOAgent, PPOConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="scenario_1_single_striker")
    p.add_argument("--total_steps", type=int, default=500_000)
    p.add_argument("--rollout_steps", type=int, default=2048)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--minibatch_size", type=int, default=512)
    p.add_argument("--load", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--log_interval", type=int, default=10, help="Log every N updates")
    p.add_argument("--save_interval", type=int, default=50, help="Save checkpoint every N updates")
    p.add_argument("--device", default="cpu")
    # TensorBoard
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--tb_dir", default="runs", help="TensorBoard log directory")
    # Reward config overrides
    p.add_argument("--ball_progress_euclidean", action="store_true", help="Use Euclidean ball progress")
    p.add_argument("--ball_progress_scale", type=float, default=None)
    p.add_argument("--wall_bounce_penalty", type=float, default=None)
    p.add_argument("--time_penalty", type=float, default=None)
    p.add_argument("--touch_reward", type=float, default=None)
    p.add_argument("--idle_penalty", type=float, default=None)
    p.add_argument("--goal_reward", type=float, default=None)
    # Ball approach shaping
    p.add_argument("--ball_approach", action="store_true", help="Add potential-based ball-approach shaping")
    p.add_argument("--approach_scale", type=float, default=3.0, help="Ball approach reward scale")
    # Env config
    p.add_argument("--no_dribble", action="store_true", help="Kick-only mode (no ball carrying)")
    p.add_argument("--fix_attack_dir", action="store_true", help="Disable attack direction randomization")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> SimulatorConfig:
    cfg = make_default_config()
    if args.ball_progress_euclidean:
        cfg.rewards.ball_progress_euclidean = True
    if args.ball_progress_scale is not None:
        cfg.rewards.ball_progress_scale = args.ball_progress_scale
    if args.wall_bounce_penalty is not None:
        cfg.rewards.wall_bounce_penalty = args.wall_bounce_penalty
    if args.time_penalty is not None:
        cfg.rewards.time_penalty = args.time_penalty
    if args.touch_reward is not None:
        cfg.rewards.touch_reward = args.touch_reward
    if args.idle_penalty is not None:
        cfg.rewards.idle_penalty = args.idle_penalty
    if args.goal_reward is not None:
        cfg.rewards.goal_reward = args.goal_reward
    if args.no_dribble:
        cfg.ball.allow_dribble = False
    if args.fix_attack_dir:
        cfg.randomization.randomize_attack_direction = False
    return cfg


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    sim_config = build_config(args)

    # TensorBoard
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_path = Path(args.tb_dir) / args.scenario
        writer = SummaryWriter(log_dir=str(tb_path))
        print(f"TensorBoard: {tb_path}")

    # Environment
    env = FootballGymEnv(
        scenario_name=args.scenario,
        config=sim_config,
        flatten_observation=True,
        canonical_observation=True,
    )
    if args.ball_approach:
        from football_rl.wrappers.ball_approach import BallApproachWrapper
        env = BallApproachWrapper(env, approach_scale=args.approach_scale)
        print(f"Ball approach shaping: scale={args.approach_scale}")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Scenario : {args.scenario}")
    print(f"Obs dim  : {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Reward config: ball_progress_scale={sim_config.rewards.ball_progress_scale}, "
          f"euclidean={sim_config.rewards.ball_progress_euclidean}, "
          f"touch={sim_config.rewards.touch_reward}, "
          f"idle={sim_config.rewards.idle_penalty}, "
          f"wall_bounce={sim_config.rewards.wall_bounce_penalty}, "
          f"time_penalty={sim_config.rewards.time_penalty}")

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

    for _ in range(num_updates):
        last_value, rollout_stats = agent.collect_rollout(env)
        update_stats = agent.update(last_value)

        total_env_steps += args.rollout_steps
        update_idx += 1

        if rollout_stats["num_episodes"] > 0:
            all_returns.append(rollout_stats["mean_ep_return"])

        if update_idx % args.log_interval == 0:
            recent = np.mean(all_returns[-20:]) if all_returns else float("nan")

            # Main metrics line
            print(
                f"Update {update_idx:4d}/{num_updates} | "
                f"steps {total_env_steps:>8,} | "
                f"return(20ep) {recent:7.3f} | "
                f"pg {update_stats['pg_loss']:6.4f} | "
                f"v {update_stats['v_loss']:6.4f} | "
                f"ent {update_stats['entropy']:5.3f} | "
                f"clip {update_stats['clip_frac']:.2f} | "
                f"kl {update_stats['approx_kl']:.4f} | "
                f"ev {update_stats['explained_var']:.2f}"
            )

            # Reward breakdown
            rbt = rollout_stats.get("reward_term_totals", {})
            if rbt:
                parts = [f"{k.replace('RewardTerm','').replace('Term','')}: {v/args.rollout_steps:.4f}"
                         for k, v in rbt.items() if abs(v) > 1e-8]
                if parts:
                    print(f"  rewards/step: {', '.join(parts)}")

            # Event counts
            evts = rollout_stats.get("event_counts", {})
            if evts:
                evt_parts = [f"{k}: {v}" for k, v in sorted(evts.items())]
                print(f"  events: {', '.join(evt_parts)}")

        # TensorBoard logging
        if writer:
            step = total_env_steps
            writer.add_scalar("rollout/mean_ep_return", rollout_stats["mean_ep_return"], step)
            writer.add_scalar("rollout/mean_ep_length", rollout_stats["mean_ep_length"], step)
            writer.add_scalar("rollout/num_episodes", rollout_stats["num_episodes"], step)
            writer.add_scalar("losses/pg_loss", update_stats["pg_loss"], step)
            writer.add_scalar("losses/v_loss", update_stats["v_loss"], step)
            writer.add_scalar("losses/entropy", update_stats["entropy"], step)
            writer.add_scalar("diagnostics/clip_frac", update_stats["clip_frac"], step)
            writer.add_scalar("diagnostics/approx_kl", update_stats["approx_kl"], step)
            writer.add_scalar("diagnostics/explained_var", update_stats["explained_var"], step)
            writer.add_scalar("diagnostics/grad_norm", update_stats["grad_norm"], step)
            writer.add_scalar("diagnostics/adv_mean", update_stats["adv_mean"], step)
            writer.add_scalar("diagnostics/adv_std", update_stats["adv_std"], step)
            # Per-term rewards (averaged per step)
            for term_name, total_val in rollout_stats.get("reward_term_totals", {}).items():
                writer.add_scalar(f"rewards/{term_name}", total_val / args.rollout_steps, step)
            # Event rates
            for evt_name, count in rollout_stats.get("event_counts", {}).items():
                writer.add_scalar(f"events/{evt_name}", count, step)
            # Action stats per dimension
            action_labels = ["move_x", "move_y", "kick_x", "kick_y", "kick_power", "sprint"]
            for i, label in enumerate(action_labels):
                writer.add_scalar(f"actions/mean_{label}", update_stats["action_mean"][i], step)
                writer.add_scalar(f"actions/std_{label}", update_stats["action_std"][i], step)
                writer.add_scalar(f"policy/std_{label}", update_stats["policy_std"][i], step)

        if update_idx % args.save_interval == 0:
            ckpt_path = save_dir / f"ppo_{args.scenario}_step{total_env_steps}.pt"
            agent.save(ckpt_path)
            print(f"  → saved {ckpt_path}")

    # Final checkpoint
    final_path = save_dir / f"ppo_{args.scenario}_final.pt"
    agent.save(final_path)
    print(f"\nTraining done. Final checkpoint: {final_path}")
    if writer:
        writer.close()
    env.close()


if __name__ == "__main__":
    main()
