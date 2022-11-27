import numpy as np
import torch as t
import random
from typing import Optional, Union
import gym
import gym.spaces
import gym.envs.registration
from gym.utils import seeding
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
from dataclasses import asdict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

from w6d2.solutions import Norvig, policy_eval_exact, policy_eval_numerical, policy_improvement

gamma = 0.9
norvig = Norvig(-0.04)

pi_up = np.zeros(12, dtype=int)  # always go up
pi_caution = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3], dtype=int)  # cautiously walk towards +1
pi_risky = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 3], dtype=int)  # shortest path to +1
pi_suicidal = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0], dtype=int)  # shortest path to +1 or -1
pi_immortal = np.array([2, 3, 3, 0, 1, 0, 2, 0, 2, 3, 3, 3], dtype=int)  # hide behind wall

policies = np.stack((pi_caution, pi_risky, pi_suicidal, pi_immortal, pi_up))

values = np.array(list(map(lambda pi: policy_eval_exact(norvig, pi, gamma), policies)))


def linear_schedule(current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int) -> float:
    """Return the appropriate epsilon for the current step.
    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).
    """
    "SOLUTION"
    duration = exploration_fraction * total_timesteps
    slope = (end_e - start_e) / duration
    return max(slope * current_step + start_e, end_e)

def test_linear_schedule(linear_schedule):
    expected = t.tensor([linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
        for step in range(500)])
    actual = t.tensor([linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500) 
        for step in range(500)])
    assert expected.shape == actual.shape
    np.testing.assert_allclose(expected, actual)

def test_policy_eval(policy_eval, exact=False):

    # try a handful of random policies

    for pi in policies:
        pi = np.random.randint(norvig.num_actions, size=(norvig.num_states,))
        if exact:
            expected = policy_eval_exact(norvig, pi, gamma=0.9)
        else:
            expected = policy_eval_numerical(norvig, pi, gamma=0.9, eps=1e-8)
        actual = policy_eval(norvig, pi, gamma=0.9)
        assert actual.shape == (norvig.num_states,)
        t.testing.assert_close(t.tensor(expected), t.tensor(actual))


def test_policy_improvement(policy_improvement):
    for v in values:
        expected = policy_improvement(norvig, v, gamma)
        actual = policy_improvement(norvig, v, gamma)
        t.testing.assert_close(t.tensor(expected), t.tensor(actual))


def test_find_optimal_policy(find_optimal_policy):

    # can't easily compare policies directly
    # as optimal policy is not unique
    # compare value functions instead

    gamma = 0.99

    env_mild = Norvig(-0.02)
    env_painful = Norvig(-0.1)
    env_hell = Norvig(-10)
    env_heaven = Norvig(10)
    enviros = [env_mild, env_painful, env_hell, env_heaven]

    for i in range(4):
        expected_pi_opt = policies[i]
        actual_pi_opt = find_optimal_policy(enviros[i], gamma)
        # print("Expected Policy")
        # print(enviros[i].render(expected_pi_opt))  # maybe have it print the policy in a nice way?
        # print(enviros[i].render(actual_pi_opt))
        val1 = policy_eval_exact(norvig, expected_pi_opt, gamma)
        val2 = policy_eval_exact(norvig, actual_pi_opt, gamma)
        t.testing.assert_close(t.tensor(val1), t.tensor(val2))


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    """Return a function that returns an environment after setting up boilerplate."""
    
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk
