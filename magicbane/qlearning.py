import numpy as np
from .direction import Direction
from itertools import product

EMPTY = 2359
FLOOR = 2379
# I'm guessing about 63-65
WALLS = set([2360, 2361, 2362, 2363, 2364, 2365])
MAP_SIZE = (21, 79)


def not_traversable(point, glyphs, seen):
    glyph = glyphs[point]
    if glyph in WALLS:
        return True
    if glyph == EMPTY and seen[point]:
        return True
    return False


def get_player_loc(obs):
    # blstats shows it in x,y so reverse
    return obs["blstats"][:2][::-1]


class NLEState:

    def __init__(self, radius=4):
        self.radius = radius
        self.seen = np.zeros(MAP_SIZE)
        self.state_lookup = {state: i for i, state in enumerate(self.enumerate_states(radius))}

    def reset(self):
        self.seen = np.zeros(MAP_SIZE)

    @staticmethod
    def enumerate_states(radius):
        dists = list(range(1, radius+1))
        for state in product(dists, dists, dists, dists):
            yield tuple(state)

    def wall_search(self, loc, glyphs, direction):
        for r in range(1, self.radius+1):
            loc = direction.apply(loc)
            stopped = loc[0] < 0 or loc[0] >= MAP_SIZE[0] or loc[1] < 0 or loc[1] >= MAP_SIZE[1]
            if not stopped:
                stopped = not_traversable(loc, glyphs, self.seen)
            if stopped:
                return r
        return r

    def construct_state_from_obs(self, obs):
        player = get_player_loc(obs)
        self.seen[player[0]-1:player[0]+2, player[1]-1:player[1]+2] = 1

        return tuple(self.wall_search(player, obs["glyphs"], d) for d in Direction.cardinal_directions())


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
