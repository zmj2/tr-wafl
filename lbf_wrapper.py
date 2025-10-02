import numpy as np
import gymnasium as gym
import lbforaging
from gymnasium import make as gym_make
from gymnasium.spaces import Discrete, Box

class LBFParallelLike:
    """
    reset() -> (obs_dict, info_dict)
    step(action_dict) -> (obs_dict, rew_dict, term_dict, trunc_dict, info_dict)
    """
    metadata = {"name": "lbf_parallel_like"}

    def __init__(self, grid=(8, 8), players=3, foods=3, coop=False,
                 max_episode_steps=50, seed=1, observation="vector"):
        self.players = players
        self.agent_ids = [f"agent_{i}" for i in range(players)]
        self.grid = grid
        self.foods = foods
        self.coop = coop
        self.max_episode_steps = max_episode_steps
        self.seed = seed

        coop_suffix = "-coop" if coop else ""
        # eg: Foraging-10x10-3p-3f-coop-v3
        self.env_id = f"Foraging-{grid[0]}x{grid[1]}-{players}p-{foods}f{coop_suffix}-v3"

        self.env = gym_make(self.env_id, max_episode_steps=max_episode_steps, render_mode=None)
        self.np_random = np.random.RandomState(seed)

        obss, _ = self.env.reset(seed=seed)
        assert isinstance(obss, (list, tuple)) and len(obss) == players
        self._obs_dim = int(np.asarray(obss[0], dtype=np.float32).shape[0])
        self._action_space = Discrete(6) # LBF: 6 (NONE/N/S/W/E/LOAD)
        self._obs_space = Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        self.possible_agents = list(self.agent_ids)

    def observation_space(self, agent):
        return self._obs_space
    
    def action_space(self, agent):
        return self._action_space
    
    def reset(self, seed=None):
        if seed is not None:
            obss, info = self.env.reset(seed=seed)
        else:
            obss, info = self.env.reset()
        obs_dict = {aid: np.asarray(obss[i], dtype=np.float32) for i, aid in enumerate(self.agent_ids)}
        info_dict = {aid: {} for aid in self.agent_ids}
        return obs_dict, info_dict
    
    def step(self, action_dict):
        actions = [int(action_dict[aid]) for aid in self.agent_ids]
        obss, rewards, terminated, truncated, info = self.env.step(actions)

        obs_dict = {aid: np.asarray(obss[i], dtype=np.float32) for i, aid in enumerate(self.agent_ids)}
        rew_dict = {aid: float(rewards[i]) for i, aid in enumerate(self.agent_ids)}
        term_dict = {aid: bool(terminated) for aid in self.agent_ids}
        trunc_dict = {aid: bool(truncated) for aid in self.agent_ids}
        info_dict = {aid: {} for aid in self.agent_ids}
        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict

    def close(self):
        self.env.close()