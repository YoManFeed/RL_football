from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from football_rl.training.policies import SharedEncoderRolePolicy
from football_rl.wrappers.gym_env import FootballGymEnv


@dataclass(slots=True)
class CurriculumStage:
    name: str
    scenario_name: str
    role: str = "striker"
    load_checkpoint: str | None = None
    freeze_encoder: bool = False


class CurriculumRunner:
    def __init__(self, stages: list[CurriculumStage], workdir: str | Path = "./artifacts"):
        self.stages = stages
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def make_env(self, stage: CurriculumStage) -> FootballGymEnv:
        return FootballGymEnv(scenario_name=stage.scenario_name, render_mode=None, canonical_observation=True, flatten_observation=True)

    def make_policy(self, observation_dim: int, stage: CurriculumStage) -> SharedEncoderRolePolicy:
        policy = SharedEncoderRolePolicy(observation_dim=observation_dim)
        if stage.load_checkpoint:
            policy.load_encoder_from(stage.load_checkpoint)
        if stage.freeze_encoder:
            policy.freeze_encoder()
        return policy

    def run_stub(self, seed: int = 7, steps_per_stage: int = 64) -> list[str]:
        checkpoints: list[str] = []
        for stage in self.stages:
            env = self.make_env(stage)
            obs, _ = env.reset(seed=seed)
            obs_vec = obs if obs.ndim == 1 else obs.reshape(-1)
            policy = self.make_policy(int(obs_vec.shape[0]), stage)
            total_reward = 0.0
            for _ in range(steps_per_stage):
                action = policy.act(obs_vec, stage.role)
                obs, reward, terminated, truncated, _ = env.step(action)
                obs_vec = obs if obs.ndim == 1 else obs.reshape(-1)
                total_reward += float(reward)
                if terminated or truncated:
                    obs, _ = env.reset(seed=seed)
                    obs_vec = obs if obs.ndim == 1 else obs.reshape(-1)
            ckpt = str(self.workdir / f"{stage.name}.npz")
            policy.save(ckpt)
            checkpoints.append(ckpt)
            env.close()
        return checkpoints


def make_default_curriculum() -> list[CurriculumStage]:
    return [
        CurriculumStage(name="stage_a_contact", scenario_name="scenario_1_single_striker"),
        CurriculumStage(name="stage_b_empty_goal", scenario_name="scenario_1_single_striker"),
        CurriculumStage(name="stage_c_moving_wall", scenario_name="scenario_2_moving_wall"),
        CurriculumStage(name="stage_d_pass_then_goal", scenario_name="scenario_3_pass_then_goal"),
        CurriculumStage(name="stage_e_goalkeeper", scenario_name="scenario_5_pass_goalkeeper"),
        CurriculumStage(name="stage_f_self_play", scenario_name="scenario_7_full_match"),
    ]
