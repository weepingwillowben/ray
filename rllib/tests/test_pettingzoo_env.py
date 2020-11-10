import unittest
from copy import deepcopy
import numpy as np
from gym.spaces import Box, Discrete

import ray
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.agents.registry import get_agent_class

from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector


class DummyEnv(AECEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        observations = {
            "a{}".format(idx): np.zeros([32], dtype=np.float32) + idx
            for idx in range(2)
        }
        observation_spaces = {
            "a{}".format(idx): Box(
                low=np.float32(0.0), high=np.float32(10.0), shape=[32])
            for idx in range(2)
        }
        action_spaces = {"a{}".format(idx): Discrete(5) for idx in range(2)}

        self._observations = observations
        self.observation_spaces = observation_spaces

        self.agents = [x for x in observation_spaces.keys()]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.action_spaces = action_spaces

        self.steps = 0

    def seed(self, seed=None):
        pass

    def observe(self, agent):
        return self._observations[agent]

    def step(self, action, observe=True):
        if self.dones[self.agent_selection]:
            self._was_done_step(action)
        self._cumulative_rewards[self.agent_selection] = 0

        self.agent_selection = self._agent_selector.next()
        self.steps += 1
        if self.steps > 10:
            self.dones = {a: True for a in self.agents}

        self._accumulate_rewards()
        self._dones_step_first()

    def reset(self, observe=True):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {a: 1 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def close(self):
        pass


class TestPettingZooEnv(unittest.TestCase):
    def setUp(self) -> None:
        ray.init()

    def tearDown(self) -> None:
        ray.shutdown()

    def test_pettingzoo_env(self):
        register_env("dummy_env", lambda _: PettingZooEnv(DummyEnv()))

        agent_class = get_agent_class("PPO")

        config = deepcopy(agent_class._default_config)

        test_env = PettingZooEnv(DummyEnv())
        obs_space = test_env.observation_space
        act_space = test_env.action_space
        test_env.close()

        config["multiagent"] = {
            "policies": {
                # the first tuple value is None -> uses default policy
                "av": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda agent_id: "av"
        }

        config["log_level"] = "DEBUG"
        config["num_workers"] = 0
        config["rollout_fragment_length"] = 30
        config["train_batch_size"] = 200
        config["horizon"] = 200  # After n steps, force reset simulation
        config["no_done_at_end"] = False

        agent = agent_class(env="dummy_env", config=config)
        agent.train()


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
