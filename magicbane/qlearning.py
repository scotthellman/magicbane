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
            # q goes negative on NCHain-v0 which has only positive rewards, this is obviously not right
            td_term = last_reward + self.gamma * future_reward - self.Q[self.last_state, self.last_action]
            trace_term = self.eligibility*(td_term - self.Q)
            self.Q += self.alpha*trace_term

        if np.random.random() > self.epsilon:
            next_action = np.random.randint(self.Q.shape[1])

            # can't trace past exploration
            self.eligibility *= 0
        else:
            next_action = np.argmax(self.Q[new_state])
            self.eligibility *= self.lam * self.gamma

        self.eligibility[new_state, next_action] = 1
        self.last_state = new_state
        self.last_action = next_action
        return next_action


if __name__ == "__main__":
    import gym


    env = gym.make("NChain-v0")
    learner = QLearner(env.observation_space.n, env.action_space.n, epsilon=0.75)
    obs = env.reset()
    reward = None
    done = False
    for i in range(100):
        if i % 25 == 0:
            print(learner.Q)
        action = learner.step(obs, reward)
        if done:
            break
        obs, reward, done, info = env.step(action)
    env.close()
