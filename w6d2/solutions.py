# %%
from gettext import find
from typing import Optional, Union
import numpy as np
import gym
import gym.spaces
import gym.envs.registration
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw
from fancy_einsum import einsum
from einops import rearrange

MAIN = __name__ == "__main__"
Arr = np.ndarray

if MAIN:
    from w6d2 import utils

# %%
class Environment:
    def __init__(self, num_states: int, num_actions: int, start=0, terminal=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.start = start
        self.terminal = np.array([], dtype=int) if terminal is None else terminal
        (self.T, self.R) = self.build()

    def build(self):
        '''
        Constructs the T and R tensors from the dynamics of the environment.
        Outputs:
            T : (num_states, num_actions, num_states) State transition probabilities
            R : (num_states, num_actions, num_states) Reward function
        '''
        num_states = self.num_states
        num_actions = self.num_actions
        T = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                (states, rewards, probs) = self.dynamics(s, a)
                (all_s, all_r, all_p) = self.out_pad(states, rewards, probs)
                T[s, a, all_s] = all_p
                R[s, a, all_s] = all_r
        return (T, R)

    def dynamics(self, state: int, action: int) -> tuple[Arr, Arr, Arr]:
        '''
        Computes the distribution over possible outcomes for a given state
        and action.
        Inputs:
            state : int (index of state)
            action : int (index of action)
        Outputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        '''
        raise NotImplementedError

    def render(pi: Arr):
        '''
        Takes a policy pi, and draws an image of the behavior of that policy,
        if applicable.
        Inputs:
            pi : (num_actions,) a policy
        Outputs:
            None
        '''
        raise NotImplementedError

    def out_pad(self, states: Arr, rewards: Arr, probs: Arr):
        '''
        Inputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        Outputs:
            states  : (num_states,) all the next states
            rewards : (num_states,) rewards for each next state transition
            probs   : (num_states,) likelihood of each state-reward pair (including
                           probability zero outcomes.)
        '''
        out_s = np.arange(self.num_states)
        out_r = np.zeros(self.num_states)
        out_p = np.zeros(self.num_states)
        for i in range(len(states)):
            idx = states[i]
            out_r[idx] += rewards[i]
            out_p[idx] += probs[i]
        return (out_s, out_r, out_p)

# %%

class Toy(Environment):

    def __init__(self):
        super().__init__(num_states=3, num_actions=2)

    def dynamics(self, state: int, action: int):
        (S0, SL, SR) = (0, 1, 2)
        LEFT = 0
        assert 0 <= state < self.num_states and 0 <= action < self.num_actions
        if state == S0:
            if action == LEFT:
                (next_state, reward) = (SL, 1)
            else:
                (next_state, reward) = (SR, 0)
        elif state == SL:
            (next_state, reward) = (S0, 0)
        elif state == SR:
            (next_state, reward) = (S0, 2)
        return (np.array([next_state]), np.array([reward]), np.array([1]))


# %%

if MAIN:
    toy = Toy()
    print(toy.T)
    print(toy.R)
# %%

class Norvig(Environment):
    def dynamics(self, state : int, action : int) -> tuple[Arr, Arr, Arr]:
        def state_index(state):
            assert 0 <= state[0] < self.width and 0 <= state[1] < self.height, print(state)
            pos = state[0] + state[1] * self.width
            assert 0 <= pos < self.num_states, print(state, pos)
            return pos

        pos = self.states[state]
        move = self.actions[action]

        # When in either goal state (or the wall), stay there forever, no reward
        if state in self.terminal or state in self.walls:
            return (np.array([state]), np.array([0]), np.array([1]))

        # 70% chance of moving in correct direction
        # 10% chance of moving in the other directions
        out_probs = np.zeros(self.num_actions) + 0.1  # set slippery probability
        out_probs[action] = 0.7  # probability of requested direction

        out_states = np.zeros(self.num_actions, dtype=int) + self.num_actions
        out_rewards = np.zeros(self.num_actions) + self.penalty
        new_states = [pos + x for x in self.actions]

        for i, s_new in enumerate(new_states):

            # check if left bounds of world, if so, don't move
            if not (0 <= s_new[0] < self.width and 0 <= s_new[1] < self.height):
                out_states[i] = state
                continue

            # position in bounds, lookup state index
            new_state = state_index(s_new)  # lookup state index

            # check if would run into a wall, if so, don't move
            if new_state in self.walls:
                out_states[i] = state

            # a normal movement, move to new cell
            else:
                out_states[i] = new_state

            # walking into a goal state from non-goal state
            for idx in range(len(self.terminal)):
                if new_state == self.terminal[idx]:
                    out_rewards[i] = self.goal_rewards[idx]

        return (out_states, out_rewards, out_probs)

    def render(self, pi: Arr):
        pi = pi.reshape((3, 4))
        objects = {(3, 0): "green", (1, 1): "black", (3, 1): "red"}
        img = Image.new(mode="RGB", size=(400, 300), color="white")
        draw = ImageDraw.Draw(img)
        for x in range(0, img.width+1, 100):
            draw.line([(x, 0), (x, img.height)], fill="black", width=4)
        for y in range(0, img.height+1, 100):
            draw.line([(0, y), (img.width, y)], fill="black", width=4)
        for x in range(4):
            for y in range(3):
                bounds = (50+x*100, 50+y*100)
                draw.regular_polygon((*bounds, 20), 3, rotation=-int(90*pi[y][x]), fill="black")
                if (x, y) in objects:
                    draw.regular_polygon((*bounds, 40), 4, fill=objects[(x, y)])
        img.show()

    def __init__(self, penalty=-0.04):

        self.height = 3
        self.width = 4
        self.penalty = penalty
        num_states = self.height * self.width
        num_actions = 4
        self.states = np.array([[x, y] for y in range(self.height) for x in range(self.width)])
        self.actions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])  # up, right, down, left
        self.dim = (self.height, self.width)

        # special states: tuples of state and reward
        # all other states get penalty
        start = 8
        terminal = np.array([3, 7], dtype=int)
        self.walls = np.array([5], dtype=int)
        self.goal_rewards = np.array([1.0, -1.0])

        super().__init__(num_states, num_actions, start=start, terminal=terminal)


# %%

def policy_eval_numerical(env: Environment, pi: Arr, gamma=0.99, eps=1e-08, max_iterations=10_000) -> Arr:
    '''
    Numerically evaluates the value of a given policy by iterating the Bellman equation
    Inputs:
        env: Environment
        pi : shape (num_states,) - The policy to evaluate
        gamma: float - Discount factor
        eps  : float - Tolerance
    Outputs:
        value : float (num_states,) - The value function for policy pi
    '''
    # Indexing T into an array of shape (num_states, num_states)
    states = np.arange(env.num_states)
    actions = pi
    transition_matrix = env.T[states, actions, :]
    # Same thing with R
    reward_matrix = env.R[states, actions, :]
    
    # Iterate until we get convergence
    V = np.zeros_like(pi)
    for i in range(max_iterations):
        V_new = einsum("s s_prime, s s_prime -> s", transition_matrix, reward_matrix + gamma * V)
        if np.abs(V - V_new).max() < eps:
            print(f"Converged in {i} steps.")
            return V_new
        V = V_new
    print(f"Failed to converge in {max_iterations} steps.")
    return V

# Alternate solution, which doesn't index at the start:

def policy_eval_numerical_2(env : Environment, pi : Arr, gamma=0.99, eps=1e-8) -> Arr:
    num_states = env.num_states
    T = env.T
    R = env.R
    delta = float("inf")
    value = np.zeros((num_states,))  
    
    while delta > eps:
        new_v = np.zeros_like(value)
        idx = range(num_states)
        for s in range(num_states):
            new_v[s] = np.dot(T[s, pi[s]], (R[s, pi[s]] + gamma * value))
        delta = np.abs(new_v - value).sum()
        value = np.copy(new_v)
    return value

if MAIN:
    utils.test_policy_eval(policy_eval_numerical, exact=False)

# %%

def policy_eval_exact(env: Environment, pi: Arr, gamma=0.99) -> Arr:
    '''
    Finds the exact solution to the Bellman equation.
    '''
    states = np.arange(env.num_states)
    actions = pi
    transition_matrix = env.T[states, actions, :]
    reward_matrix = env.R[states, actions, :]

    r = einsum("i j, i j -> i", transition_matrix, reward_matrix)

    mat = np.eye(env.num_states) - gamma * transition_matrix

    return np.linalg.solve(mat, r)

if MAIN:
    utils.test_policy_eval(policy_eval_exact, exact=True)

# %%

def policy_improvement(env: Environment, V: Arr, gamma=0.99) -> Arr:
    '''
    Inputs:
        env: Environment
        V  : (num_states,) value of each state following some policy pi
    Outputs:
        pi_better : vector (num_states,) of actions representing a new policy obtained via policy iteration
    '''
    states = np.arange(env.num_states)
    transition_matrix = env.T[states, :, :]
    reward_matrix = env.R[states, :, :]
    
    V_for_each_action = einsum("s a s_prime, s a s_prime -> s a", transition_matrix, reward_matrix + gamma * V)
    pi_better = V_for_each_action.argmax(-1)

    return pi_better

# Alternate solution:

def policy_improvement_2(env : Environment, V : Arr, gamma=0.99) -> Arr:
    pi_new = np.argmax(np.einsum("ijk,ijk -> ij", env.T, env.R) + gamma * np.einsum("ijk,k -> ij", env.T, V), axis=1)
    return pi_new

if MAIN:
    utils.test_policy_improvement(policy_improvement)

# %%

def find_optimal_policy(env: Environment, gamma=0.99, max_iterations=10_000):
    '''
    Inputs:
        env: environment
    Outputs:
        pi : (num_states,) int, of actions represeting an optimal policy
    '''
    pi = np.zeros(shape=env.num_states, dtype=int)

    for i in range(max_iterations):
        V = policy_eval_exact(env, pi, gamma)
        pi_new = policy_improvement(env, V, gamma)
        if np.array_equal(pi_new, pi):
            return pi_new
        else:
            pi = pi_new
    else:
        print(f"Failed to converge after {max_iterations} steps.")
        return pi

if MAIN:
    utils.test_find_optimal_policy(find_optimal_policy)
    penalty = -0.1
    norvig = Norvig(penalty)
    pi_opt = find_optimal_policy(norvig, gamma=0.99)
    norvig.render(pi_opt)

# %%