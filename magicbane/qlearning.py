import numpy as np


class QLearner:

    def __init__(self, num_states, num_actions, alpha=0.5, gamma=0.95, lam=1, epsilon=0.95):
        self.Q = np.zeros((num_states, num_actions))
        self.eligibility = np.zeros((num_states, num_actions))
        self.last_state = None
        self.last_action = None

        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon

    def step(self, new_state, last_reward):
        if last_reward is not None:
            future_reward = np.max(self.Q[new_state])
            delta = last_reward + self.gamma * future_reward - self.Q[self.last_state, self.last_action]
            trace_term = self.eligibility*delta
            self.Q += self.alpha*trace_term

        if np.random.random() > self.epsilon:
            next_action = np.random.randint(self.Q.shape[1])
            # can't trace past exploration
            self.eligibility *= 0
        else:
            next_action = np.argmax(self.Q[new_state])
            self.eligibility *= self.lam

        self.eligibility[new_state, next_action] += 1
        self.last_state = new_state
        self.last_action = next_action
        return next_action


if __name__ == "__main__":
    import gym

    env = gym.make("NChain-v0")
    learner = QLearner(env.observation_space.n, env.action_space.n, alpha=0.05, gamma=0.9, lam=0.5, epsilon=0.9)
    for episode in range(20):
        print(learner.Q)
        obs = env.reset()
        reward = None
        done = False
        for i in range(1000):
            action = learner.step(obs, reward)
            if done:
                break
            obs, reward, done, info = env.step(action)
    env.close()
