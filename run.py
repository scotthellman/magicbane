import gym
import nle
import time
from magicbane import qlearning
from magicbane import direction
from magicbane import deepq

EPISODES = 100
STEPS = 5000


if __name__ == "__main__":
    import numpy as np
    env = gym.make("NetHackScore-v0")
    obs = env.reset()  # each reset generates a new dungeon
    state_maker = qlearning.NLEState(radius=6)
    actions = [d.action() for d in direction.Direction]
    agent = deepq.DeepQAgent(len(actions), C=100)
    reward = None
    rewards = []
    for ep in range(EPISODES):
        env.reset()
        state_maker.reset()
        done = False
        for step in range(STEPS):
            state = agent.build_state(obs)
            action = actions[agent.step(state, reward, done)]
            if done:
                break
            obs, reward, done, info = env.step(action)
        env.render()
    env.close()
    print(rewards)
