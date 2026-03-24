from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ImportError:  # pragma: no cover
    class Env:
        metadata = {}

        def reset(self, *, seed: int | None = None, options=None):
            self.np_random = np.random.default_rng(seed)
            return None

        def step(self, action):
            raise NotImplementedError

        def render(self):
            return None

        def close(self):
            return None

    class Space:
        def sample(self):
            raise NotImplementedError

    class Box(Space):
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.shape = tuple(shape or self.low.shape)
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    class Dict(Space):
        def __init__(self, spaces_dict):
            self.spaces = spaces_dict

        def sample(self):
            return {key: space.sample() for key, space in self.spaces.items()}

    class ObservationWrapper(Env):
        def __init__(self, env):
            self.env = env
            self.observation_space = getattr(env, 'observation_space', None)
            self.action_space = getattr(env, 'action_space', None)

        def observation(self, observation):
            return observation

        def reset(self, *, seed: int | None = None, options=None):
            observation, info = self.env.reset(seed=seed, options=options)
            return self.observation(observation), info

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            return self.observation(observation), reward, terminated, truncated, info

        def render(self):
            return self.env.render()

        def close(self):
            return self.env.close()

    class _GymModule:
        Env = Env
        ObservationWrapper = ObservationWrapper

    class _SpacesModule:
        Box = Box
        Dict = Dict

    gym = _GymModule()
    spaces = _SpacesModule()
