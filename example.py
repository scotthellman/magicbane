import gym
import nle
import time

env = gym.make("NetHackScore-v0")
env.reset()  # each reset generates a new dungeon
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.1)
env.close()
