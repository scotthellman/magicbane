import gym
import copy
import nle
import time
from magicbane import qlearning
from magicbane import direction
from magicbane import deepq
import numpy as np
import torch

EPISODES = 20000
STEPS = 10000
model_out = "nle_deepq.params"

def nle_state_builder(obs):
    glyphs = obs["glyphs"]
    return torch.from_numpy(glyphs).int()

def frozen_state_builder(obs):
    tile_indices = {
                b"@": 0,
                b"F": 1,
                b"H": 2,
                b"G": 3,
                b"S": 4
            }
    grid = torch.zeros((8, 8), dtype=torch.long)
    for i, line in enumerate(obs):
        for j, char in enumerate(line):
            grid[i, j] = tile_indices[char]
    return grid

def get_frozen_grid(env):
    grid = copy.deepcopy(env.desc)
    row, col = env.s // env.ncol, env.s % env.ncol
    grid[row, col] = "@"
    return grid



nle_config = {
            "gym": "NetHackScore-v0",
            "map_size": (21, 79),
            "state_extractor": nle_state_builder,
            "C": 1000,
            "vocab_size": 5991,
            "kernel_size": 5
        }

# unfortunately this one requires manual intervention
# the obs returned by default is an int
# have to use get_grozen_grid to get the grid
frozen_config = {
            "gym": "FrozenLake8x8-v0",
            "map_size": (8, 8),
            "state_extractor": frozen_state_builder,
            "C": 500,
            "vocab_size": 5,
            "kernel_size": 3
        }


config = nle_config

if __name__ == "__main__":
    env = gym.make(config["gym"])
    obs = env.reset()  # each reset generates a new dungeon
    agent = deepq.DeepQAgent(env.action_space.n, C=config["C"], map_size=config["map_size"],
                             vocab_size=config["vocab_size"], kernel_size=config["kernel_size"])
    reward = None
    rewards = []
    for ep in range(EPISODES):
        env.reset()
        done = False
        total_reward = 0
        for step in range(STEPS):
            state = config["state_extractor"](obs)
            action = agent.step(state, reward, done)
            if done:
                break
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if step %1000 == 0:
                env.render()
        rewards.append(total_reward)
    env.close()
    print(rewards)
