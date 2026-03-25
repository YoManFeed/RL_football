"""Generate GIF from a trained agent checkpoint.

Usage:
    python generate_gif.py --checkpoint checkpoints/ppo_scenario_1_single_striker_final.pt --scenario scenario_1_single_striker
    python generate_gif.py --checkpoint checkpoints/ppo_scenario_2_moving_wall_final.pt --scenario scenario_2_moving_wall --output gifs/demo.gif
    python generate_gif.py --checkpoint checkpoints/ppo_scenario_1_single_striker_final.pt --scenario scenario_1_single_striker --width 640 --height 480 --fps 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))

import torch

from football_rl.wrappers.gym_env import FootballGymEnv
from football_rl.training.ppo import PPOAgent, PPOConfig
from football_rl.configs.defaults import make_default_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate GIF from a trained PPO agent checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to the checkpoint file (.pt)",
    )
    p.add_argument(
        "--scenario",
        required=True,
        type=str,
        help="Scenario name (e.g., scenario_1_single_striker, scenario_2_moving_wall)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GIF path. If not specified, auto-generated from scenario name.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=480,
        help="GIF width in pixels (default: 480)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=320,
        help="GIF height in pixels (default: 320)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=18,
        help="GIF frames per second (default: 18)",
    )
    p.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to try; best one is selected (default: 5)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for episode rendering (default: 0)",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label to display on GIF. If not specified, auto-generated.",
    )
    p.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of agent network (default: 128)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for agent (cpu or cuda) (default: cpu)",
    )
    return p.parse_args()


def add_text(frame: np.ndarray, label: str, ret: float | None = None) -> np.ndarray:
    """Burn label + return value into the frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    text = label
    if ret is not None:
        text += f"  |  return {ret:.2f}"
    # Draw stripe at bottom instead of top
    stripe_height = 20
    draw.rectangle([0, img.height - stripe_height, img.width, img.height], fill=(20, 20, 20))
    draw.text((6, img.height - stripe_height + 2), text, fill=(220, 220, 50))
    return np.array(img)


def render_episode(
    agent: PPOAgent,
    scenario: str,
    width: int,
    height: int,
    label: str,
    seed: int = 0,
) -> tuple[list[np.ndarray], float]:
    """Run episode and return (frames, episode_return)."""
    # Use same config as training for consistent rewards
    cfg = make_default_config()
    cfg.rewards.ball_progress_scale = 2.0
    cfg.rewards.touch_reward = 0.5
    cfg.ball.allow_dribble = False
    cfg.randomization.randomize_attack_direction = False
    
    env = FootballGymEnv(
        scenario_name=scenario,
        config=cfg,
        flatten_observation=True,
        canonical_observation=True,
        render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=seed)
    frames = []
    ep_ret = 0.0
    done = False

    while not done:
        raw = env.render()
        if raw is not None:
            img = Image.fromarray(raw).resize((width, height), Image.BILINEAR)
            frames.append(np.array(img))
        action, _, _ = agent.select_action(obs)
        obs, r, terminated, truncated, _ = env.step(action)
        ep_ret += r
        done = terminated or truncated

    env.close()
    stamped = [add_text(f, label, ep_ret) for f in frames]
    return stamped, ep_ret


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Save frames as animated GIF."""
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps, loop=0)
    size_kb = path.stat().st_size // 1024
    print(f"✓ Saved {path.name}")
    print(f"  Frames: {len(frames)}")
    print(f"  Size: {size_kb} KB")
    print(f"  Path: {path.resolve()}")


def main() -> None:
    args = parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load checkpoint and extract dimensions
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location=args.device)
    net_ckpt = ckpt["net"]
    
    # Extract obs_dim and action_dim from checkpoint
    obs_dim = net_ckpt["encoder.0.weight"].shape[1]  # input size of first layer
    action_dim = net_ckpt["log_std"].shape[0]  # size of log_std vector
    
    print(f"  obs_dim: {obs_dim}, action_dim: {action_dim}")

    # Load agent
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        config=PPOConfig(),
        device=args.device,
    )
    agent.load(str(checkpoint_path))
    print("✓ Agent loaded\n")

    # Generate output path if not specified
    if args.output is None:
        output_dir = Path("gifs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{args.scenario}.gif"
    else:
        output_path = Path(args.output)

    # Generate label if not specified
    if args.label is None:
        label = f"PPO  {args.scenario}"
    else:
        label = args.label

    # Render best of N episodes
    print(f"Rendering {args.num_episodes} episodes (selecting best)...")
    best_frames: list[np.ndarray] = []
    best_ret = -1e9

    for i, trial_seed in enumerate(range(args.seed, args.seed + args.num_episodes)):
        print(f"  Episode {i + 1}/{args.num_episodes} (seed={trial_seed})...", end=" ", flush=True)
        frames, ret = render_episode(agent, args.scenario, args.width, args.height, label, seed=trial_seed)
        print(f"return={ret:.2f}")
        if ret > best_ret:
            best_ret = ret
            best_frames = frames

    print(f"\n✓ Best episode return: {best_ret:.2f}\n")

    # Save GIF
    print(f"Saving GIF to {output_path}...")
    save_gif(best_frames, output_path, args.fps)


if __name__ == "__main__":
    main()
