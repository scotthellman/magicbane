import gym
import nle
import time
from magicbane import qlearning
from magicbane import direction
from magicbane import deepq
import numpy as np

EPISODES = 100
STEPS = 10000
model_out = "nle_deepq.params"


if __name__ == "__main__":
    env = gym.make("NetHackScore-v0")
    obs = env.reset()  # each reset generates a new dungeon
    state_maker = qlearning.NLEState(radius=6)
    agent = deepq.DeepQAgent(env.action_space.n, C=1000)
    reward = None
    rewards = []
    for ep in range(EPISODES):
        env.reset()
        state_maker.reset()
        done = False
        for step in range(STEPS):
            state = agent.build_state(obs)
            action = agent.step(state, reward, done)
            if done:
                break
            obs, reward, done, info = env.step(action)
        env.render()
    env.close()
    print(rewards)
