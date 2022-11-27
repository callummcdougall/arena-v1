# %%
import argparse
import os
import sys
import random
import time
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Any, List, Optional, Union, Tuple, Iterable

import gym
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from gym.spaces import Discrete, Box
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from numpy.random import Generator
import gym.envs.registration
import pandas as pd

from w6d2.utils import make_env
from w6d4 import utils

MAIN = __name__ == "__main__"
os.environ["SDL_VIDEODRIVER"] = "dummy"

# %%

class QNetwork(nn.Module):
    def __init__(self, dim_observation: int, num_actions: int, hidden_sizes: list[int] = [120, 84]):
        super().__init__()
        "SOLUTION"
        layers: list[nn.Module] = []
        layers.append(nn.Linear(dim_observation, hidden_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        "SOLUTION"
        return self.network(x)

# hardcoded solution
# class QNetwork2(nn.Module):
#     def __init__(self, dim_observation: int, num_actions: int) -> None:
#         super().__init__()
#         layer1 = nn.Linear(dim_observation, 120)
#         layer2 = nn.Linear(120, 84)
#         layer3 = nn.Linear(84, num_actions)
#         self.network = nn.Sequential(layer1, nn.ReLU(), layer2, nn.ReLU(), layer3)
#     def forward(self, x: t.Tensor) -> t.Tensor:
#         return self.network(x)

if MAIN:
    net = QNetwork(dim_observation=4, num_actions=2)
    n_params = sum(p.nelement() for p in net.parameters())
    print(net)
    print(f"Total number of parameters: {n_params}")
    print("You should manually verify network is Linear-ReLU-Linear-ReLU-Linear")
    assert n_params == 10934
    # TBD lowpri: test equal

# %%
@dataclass
class ReplayBufferSamples:
    """
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    obs: shape (sample_size, *observation_shape), dtype t.float
    actions: shape (sample_size, ) dtype t.int
    rewards: shape (sample_size, ), dtype t.float
    dones: shape (sample_size, ), dtype t.bool
    next_observations: shape (sample_size, *observation_shape), dtype t.float
    """

    observations: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    next_observations: t.Tensor


class ReplayBuffer:
    rng: Generator

    def __init__(
        self,
        buffer_size: int,
        num_actions: int,
        observation_shape: tuple,
        num_environments: int,
        seed: int,
    ):
        assert num_environments == 1, "This buffer only supports SyncVectorEnv with 1 environment inside."
        "SOLUTION"
        self.buffer_size = buffer_size
        self.num_actions = num_actions
        self.observation_shape = observation_shape
        self.num_environments = num_environments
        self.observations = np.zeros((self.buffer_size,) + self.observation_shape, dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size,) + self.observation_shape, dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.pos = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        """
        obs: shape (num_environments, *observation_shape) Observation before the action
        actions: shape (num_environments, ) the action chosen by the agent
        rewards: shape (num_environments, ) the reward after the after
        dones: shape (num_environments, ) if True, the episode ended and was reset automatically
        next_obs: shape (num_environments, *observation_shape) Observation after the action. If done is True, this should be the terminal observation, NOT the first observation of the next episode.
        """
        "SOLUTION"
        if self.pos == 0:  # It's slow to do this on every add, just check once
            assert obs.shape == (self.num_environments,) + self.observation_shape
            assert actions.shape == (self.num_environments,)
            assert rewards.shape == (self.num_environments,)
            assert dones.shape == (self.num_environments,)
            assert next_obs.shape == (self.num_environments,) + self.observation_shape
        self.observations[self.pos] = obs
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.next_observations[self.pos] = next_obs
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        """Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        """
        "SOLUTION"
        hi = self.buffer_size if self.full else self.pos
        ind = self.rng.integers(0, hi, size=sample_size)

        return ReplayBufferSamples(
            t.as_tensor(self.observations[ind], device=device),
            t.as_tensor(self.actions[ind], device=device),
            t.as_tensor(self.rewards[ind], device=device),
            t.as_tensor(self.dones[ind], device=device),
            t.as_tensor(self.next_observations[ind], device=device),
        )
        # TBD: how helpful is it to call env.normalize_obs and env.normalize_reward here?

if MAIN:
    utils.test_replay_buffer_single(ReplayBuffer)
    utils.test_replay_buffer_deterministic(ReplayBuffer)
    utils.test_replay_buffer_wraparound(ReplayBuffer)


# %%
if MAIN:
    rb = ReplayBuffer(buffer_size=256, num_actions=2, observation_shape=(4,), num_environments=1, seed=0)
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "test")])
    obs = envs.reset()
    for i in range(512):
        actions = np.array([0])
        next_obs, rewards, dones, infos = envs.step(actions)
        real_next_obs = next_obs.copy()
        for i, done in enumerate(dones):
            if done:
                real_next_obs[i] = infos[i]["terminal_observation"]
        rb.add(obs, actions, rewards, dones, next_obs)
        obs = next_obs
    sample = rb.sample(128, t.device("cpu"))

    columns = ["cart_pos", "cart_v", "pole_angle", "pole_v"]
    df = pd.DataFrame(rb.observations, columns=columns)
    df.plot(subplots=True, title = "Replay Buffer")

    df2 = pd.DataFrame(sample.observations, columns=columns)
    df2.plot(subplots=True, title = "Shuffled Replay Buffer")


# %%
def linear_schedule(current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int) -> float:
    """Return the appropriate epsilon for the current step.
    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).
    """
    "SOLUTION"
    duration = exploration_fraction * total_timesteps
    slope = (end_e - start_e) / duration
    return max(slope * current_step + start_e, end_e)


if MAIN:
    epsilons = [
        linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
        for step in range(500)
    ]

if MAIN:
    utils.test_linear_schedule(linear_schedule)
# %%
# if MAIN:
#     fig, ax = plt.subplots()
#     ax.plot(epsilons)
#     ax.set(xlabel="Step", ylabel="Epsilon")
#     fig.suptitle("Probability of Random Action")
# %%
def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv, q_network: QNetwork, rng: Generator, obs: t.Tensor, epsilon: float
) -> np.ndarray:
    """With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
    Inputs:
        envs : gym.vector.SyncVectorEnv, the family of environments to run against
        q_network : QNetwork, the network used to approximate the Q-value function
        obs : The current observation
        epsilon : exploration percentage
    Outputs:
        actions: (n_environments, ) the sampled action for each environment.
    """
    "SOLUTION"
    if rng.random() < epsilon:
        actions = rng.integers(0, envs.single_action_space.n, size = (envs.num_envs,))
    else:
        actions = q_network(obs).argmax(dim=-1).cpu().numpy()
    assert actions.shape == (envs.num_envs,)
    return actions 
    # if rng.random() < epsilon:
    #     # TBD: can't we use envs.action_space.sample()?
    #     actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    # else:
    #     actions = torch.argmax(q_network(obs), dim=-1).cpu().numpy()
    # assert actions.shape == (envs.num_envs,)
    # return actions

if MAIN:
    utils.test_epsilon_greedy_policy(epsilon_greedy_policy)

# %%

ObsType = np.ndarray
ActType = int


class Probe1(gym.Env):
    """One action, observation of [0.0], one timestep long, +1 reward.
    We expect the agent to rapidly learn that the value of the constant [0.0] observation is +1.0. Note we're using a continuous observation space for consistency with CartPole.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([0]), np.array([0]))
        self.action_space = Discrete(1)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        return np.array([0]), 1.0, True, {}

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        if return_info:
            return np.array([0.0]), {}
        return np.array([0.0])


gym.envs.registration.register(id="Probe1-v0", entry_point=Probe1)

if MAIN:
    env = gym.make("Probe1-v0")
    assert env.observation_space.shape == (1,)
    assert env.action_space.shape == ()

"""
### Additional Probe Environments
Feel free to skip ahead for now, and implement these as needed to debug your model. 
"""
# %%
class Probe2(gym.Env):
    """One action, observation of [-1.0] or [+1.0], one timestep long, reward equals observation.
    We expect the agent to rapidly learn the value of each observation is equal to the observation.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        "SOLUTION"
        super().__init__()
        self.observation_space = Box(np.array([-1.0]), np.array([+1.0]))
        self.action_space = Discrete(1)
        self.seed()
        self.reset()
        self.reward = None

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        "SOLUTION"
        assert self.reward is not None
        return np.array([0]), self.reward, True, {}

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        "SOLUTION"
        super().reset(seed=seed)
        self.reward = 1.0 if self.np_random.random() < 0.5 else -1.0
        if return_info:
            return np.array([self.reward]), {}
        return np.array([self.reward])


gym.envs.registration.register(id="Probe2-v0", entry_point=Probe2)
# %%
class Probe3(gym.Env):
    """One action, [0.0] then [1.0] observation, two timesteps, +1 reward at the end.
    We expect the agent to rapidly learn the discounted value of the initial observation.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        "SOLUTION"
        super().__init__()
        self.observation_space = Box(np.array([-0.0]), np.array([+1.0]))
        self.action_space = Discrete(1)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        "SOLUTION"
        self.n += 1
        if self.n == 1:
            return np.array([1.0]), 0.0, False, {}
        elif self.n == 2:
            return np.array([0]), 1.0, True, {}
        raise ValueError(self.n)

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        "SOLUTION"
        super().reset(seed=seed)
        self.n = 0
        if return_info:
            return np.array([0.0]), {}
        return np.array([0.0])


gym.envs.registration.register(id="Probe3-v0", entry_point=Probe3)
# %%
class Probe4(gym.Env):
    """Two actions, [0.0] observation, one timestep, reward is -1.0 or +1.0 dependent on the action.
    We expect the agent to learn to choose the +1.0 action.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        "SOLUTION"
        super().__init__()
        self.observation_space = Box(np.array([-0.0]), np.array([+0.0]))
        self.action_space = Discrete(2)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        "SOLUTION"
        reward = 1.0 if action == 1 else -1.0
        return np.array([0.0]), reward, True, {}

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        "SOLUTION"
        super().reset(seed=seed)
        if return_info:
            return np.array([0.0]), {}
        return np.array([0.0])


gym.envs.registration.register(id="Probe4-v0", entry_point=Probe4)

# %%
class Probe5(gym.Env):
    """Two actions, random 0/1 observation, one timestep, reward is 1 if action equals observation otherwise -1.
    We expect the agent to learn to match its action to the observation.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        "SOLUTION"
        super().__init__()
        self.observation_space = Box(np.array([-1.0]), np.array([+1.0]))
        self.action_space = Discrete(2)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        "SOLUTION"
        reward = 1.0 if action == self.obs else -1.0
        return np.array([-1.0]), reward, True, {}

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        "SOLUTION"
        super().reset(seed=seed)
        self.obs = 0 if self.np_random.random() < 0.5 else 1
        if return_info:
            return np.array([self.obs], dtype=float), {}
        return np.array([self.obs], dtype=float)


gym.envs.registration.register(id="Probe5-v0", entry_point=Probe5)

# %%
@dataclass
class DQNArgs:
    exp_name: str = os.path.basename(globals().get("__file__", "DQN_implementation").rstrip(".py"))
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "CartPoleDQN"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 0.00025
    buffer_size: int = 10000
    gamma: float = 0.99
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10

arg_help_strings = {
    "exp_name": "the name of this experiment",
    "seed": "seed of the experiment",
    "torch_deterministic": "if toggled, " "`torch.backends.cudnn.deterministic=False`",
    "cuda": "if toggled, cuda will be enabled by default",
    "track": "if toggled, this experiment will be tracked with Weights and Biases",
    "wandb_project_name": "the wandb's project name",
    "wandb_entity": "the entity (team) of wandb's project",
    "capture_video": "whether to capture videos of the agent performances (check " "out `videos` folder)",
    "env_id": "the id of the environment",
    "total_timesteps": "total timesteps of the experiments",
    "learning_rate": "the learning rate of the optimizer",
    "buffer_size": "the replay memory buffer size",
    "gamma": "the discount factor gamma",
    "target_network_frequency": "the timesteps it takes to update the target " "network",
    "batch_size": "the batch size of samples from the replay memory",
    "start_e": "the starting epsilon for exploration",
    "end_e": "the ending epsilon for exploration",
    "exploration_fraction": "the fraction of `total-timesteps` it takes from " "start-e to go end-e",
    "learning_starts": "timestep to start learning",
    "train_frequency": "the frequency of training"
}
toggles = ["torch_deterministic", "cuda", "track", "capture_video"]

def parse_args(arg_help_strings=arg_help_strings, toggles=toggles) -> DQNArgs:
    parser = argparse.ArgumentParser()
    for name, field in DQNArgs.__dataclass_fields__.items():
        flag = "--" + name.replace("_", "-")
        type_function = field.type if field.type != bool else lambda x: bool(strtobool(x))
        toggle_kwargs = {"nargs": "?", "const": True} if name in toggles else {}
        parser.add_argument(
            flag, type=type_function, default=field.default, help=arg_help_strings[name], **toggle_kwargs
        )
    return DQNArgs(**vars(parser.parse_args()))
# %%

def setup(args: DQNArgs) -> Tuple[str, SummaryWriter, np.random.Generator, t.device, gym.vector.SyncVectorEnv]:
    """Helper function to set up useful variables for the DQN implementation"""
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic  # type: ignore

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"

    return run_name, writer, rng, device, envs

def log(
    writer: SummaryWriter,
    start_time: float,
    step: int,
    predicted_q_vals: t.Tensor,
    loss: Union[float, t.Tensor],
    infos: Iterable[dict],
    epsilon: float,
):
    """Helper function to write relevant info to TensorBoard logs, and print some things to stdout"""
    if step % 100 == 0:
        writer.add_scalar("losses/td_loss", loss, step)
        writer.add_scalar("losses/q_values", predicted_q_vals.mean().item(), step)
        writer.add_scalar("charts/SPS", int(step / (time.time() - start_time)), step)
        if step % 10000 == 0:
            print("SPS:", int(step / (time.time() - start_time)))

    for info in infos:  # type: ignore
        if "episode" in info.keys():
            print(f"global_step={step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], step)
            writer.add_scalar("charts/epsilon", epsilon, step)
            break
# %%

def train_dqn(args: DQNArgs):
    run_name, writer, rng, device, envs = setup(args)

    if "SOLUTION":
        obs_shape = envs.single_observation_space.shape
        num_observations = np.array(obs_shape, dtype=int).prod()
        num_actions = envs.single_action_space.n
        q_network = QNetwork(num_observations, num_actions).to(device)
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

        target_network = QNetwork(num_observations, num_actions).to(device)
        target_network.load_state_dict(q_network.state_dict())

        rb = ReplayBuffer(args.buffer_size, num_actions, envs.single_observation_space.shape, len(envs.envs), args.seed)
    else:
        """YOUR CODE: Create your Q-network, Adam optimizer, and replay buffer here."""

    start_time = time.time()
    obs = envs.reset()
    for step in range(args.total_timesteps):
        if "SOLUTION":
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(step, args.start_e, args.end_e, args.exploration_fraction, args.total_timesteps)
            actions = epsilon_greedy_policy(envs, q_network, rng, torch.Tensor(obs).to(device), epsilon)
            assert actions.shape == (len(envs.envs),)
            next_obs, rewards, dones, infos = envs.step(actions)
        else:
            """YOUR CODE: Sample actions according to the epsilon greedy policy using the linear schedule for epsilon, and then step the environment"""

        """Boilerplate to handle the terminal observation case"""
        real_next_obs = next_obs.copy()
        for i, done in enumerate(dones):
            if done:
                real_next_obs[i] = infos[i]["terminal_observation"]
        rb.add(obs, actions, rewards, dones, next_obs)
        obs = next_obs


        if step > args.learning_starts and step % args.train_frequency == 0:
            if "SOLUTION":
                data = rb.sample(args.batch_size, device)
                with torch.no_grad():
                    target_max = target_network(data.next_observations).max(dim=1).values
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                # NOTE: we asserted the action space was discrete, so this is fine
                gather_arg = rearrange(data.actions, "a -> a 1")
                predicted_q_vals = q_network(data.observations).gather(1, gather_arg).squeeze()  # (batch, num_actions)
                loss = F.mse_loss(td_target, predicted_q_vals)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % args.target_network_frequency == 0:
                    target_network.load_state_dict(q_network.state_dict())
            else:
                """YOUR CODE: Sample from the replay buffer, compute the TD target, compute TD loss, and perform an optimizer step."""

            log(writer, start_time, step, predicted_q_vals, loss, infos, epsilon)


    """If running one of the Probe environments, will test if the learned q-values are
    sensible after training. Useful for debugging."""
    # TBD: tolerances depend how long you run
    # probably we'd rather run for less time and have looser tolerances.
    # Maybe these should be proper tests?
    if args.env_id == "Probe1-v0":
        batch = t.tensor([[0.0]]).to(device)
        value = q_network(batch)
        print("Value: ", value)
        expected = t.tensor([[1.0]]).to(device)
        t.testing.assert_close(value, expected, 1e-4)
    elif args.env_id == "Probe2-v0":
        batch = t.tensor([[-1.0], [+1.0]]).to(device)
        value = q_network(batch)
        print("Value:", value)
        expected = batch
        t.testing.assert_close(value, expected, 1e-4)
    elif args.env_id == "Probe3-v0":
        batch = t.tensor([[0.0], [1.0]]).to(device)
        value = q_network(batch)
        print("Value: ", value)
        expected = t.tensor([[args.gamma], [1.0]])
        t.testing.assert_close(value, expected, 1e-4)
    elif args.env_id == "Probe4-v0":
        batch = t.tensor([[0.0]]).to(device)
        value = q_network(batch)
        expected = t.tensor([[-1.0, 1.0]]).to(device)
        print("Value: ", value)
        t.testing.assert_close(value, expected, 1e-4)
    elif args.env_id == "Probe5-v0":
        batch = t.tensor([[0.0], [1.0]]).to(device)
        value = q_network(batch)
        expected = t.tensor([[1.0, -1.0], [-1.0, 1.0]]).to(device)
        print("Value: ", value)
        t.testing.assert_close(value, expected, 1e-4)

    envs.close()
    writer.close()

if MAIN:
    if "ipykernel_launcher" in os.path.basename(sys.argv[0]):
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead: python {os.path.basename(filename)} --help")
        args = DQNArgs()
    else:
        args = parse_args()
    train_dqn(args)

# %%