import gym
import nle
import time
from magicbane import qlearning
from magicbane import direction

EPISODES = 100
STEPS = 5000


if __name__ == "__main__":
    import numpy as np
    env = gym.make("NetHackScore-v0")
    obs = env.reset()  # each reset generates a new dungeon
    state_maker = qlearning.NLEState(radius=6)
    actions = [d.action() for d in direction.Direction]
    agent = qlearning.QLearner(len(state_maker.state_lookup), len(actions),
                               alpha=0.05, gamma=0.9, lam=0.75, epsilon=0.9)
    reward = None
    rewards = []
    for ep in range(EPISODES):
        env.reset()
        state_maker.reset()
        done = False
        total_reward = 0
        for step in range(STEPS):
            if reward is not None:
                amount_seen = np.sum(state_maker.seen)
            agent_obs = state_maker.construct_state_from_obs(obs)
            if reward is not None:
                reward = np.sum(state_maker.seen) - amount_seen
                total_reward += reward
            obs_number = state_maker.state_lookup[agent_obs]
            action = actions[agent.step(obs_number, reward)]
            if done:
                break
            obs, reward, done, info = env.step(action)
        rewards.append(total_reward)
        env.render()
    env.close()
    print(rewards)
