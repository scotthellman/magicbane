import gym
import pickle
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
            "gym": "NetHackChallenge-v0",
            "map_size": (21, 79),
            "state_extractor": nle_state_builder,
            "C": 10000,
            "memory_size": 20000,
            "vocab_size": 5991,
            "kernel_size": 5,
            "epsilon_decay_window": 100000
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
    agent = deepq.DeepQAgent(env.action_space.n, **config)
    reward = None
    rewards = []
    prev_obs = ""
    replay_file = "/Users/scott/repos/magicbane/nle_data/play_data/frames.pkl"
    with open(replay_file, "rb") as f:
        replay_frames = pickle.load(f)
    print(f"agent initialized with {len(agent.memory)} memory")
    for frame in replay_frames:
        converted_frame = [torch.from_numpy(frame[0]).long(), frame[1], frame[2], torch.from_numpy(frame[3]).long(), frame[4]]
        converted_frame[0] = converted_frame[0].unsqueeze(0)
        converted_frame[3] = converted_frame[3].unsqueeze(0)
        agent.memory.insert(tuple(converted_frame))
    print(f"expert replay puts us at {len(agent.memory)}")
    for ep in range(EPISODES):
        env.reset()
        done = False
        total_reward = 0
        for step in range(STEPS):
            state = config["state_extractor"](obs)
            action = agent.step(state, reward, done)
            if done:
                print(prev_obs)
                break
            prev_obs = env.render(mode="ansi")
            obs, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        with open("model.pkl", "wb") as f:
            # TODO: in the long run this is the wrong way to save
            pickle.dump(agent, f)
    env.close()
    print(rewards)
