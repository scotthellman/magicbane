#https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F
from .datastructures import CircularBuffer
import numpy as np

MAP_SIZE = (21, 79)

MAX_GLYPH = 5991


class DeepQAgent:

    def __init__(self, num_actions, epsilon=0.1, gamma=0.99, C=1000,
                 memory_size=2000, lr=0.0001, minibatch_size=8):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.C = C
        self.memory = CircularBuffer(memory_size)
        self.Q = DeepQNetwork()
        self.target_Q = DeepQNetwork(num_actions)
        self.minibatch_size = minibatch_size
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.reset()

    def reset(self):
        self.previous_state = None
        self.previous_action = None
        self.steps = 0
        self.update_target_Q()

    def update_target_Q(self):
        self.target_Q.load_state_dict(self.Q.state_dict())


    def step(self, state, reward, terminal_state):
        # TODO: handle unfull memory
        self.steps += 1
        if self.steps == self.C:
            self.steps = 0
            self.update_target_Q()
        self.memory.insert((self.previous_state, self.previous_action, reward, state, terminal_state))
        if len(self.memory) > 2*self.minibatch_size:
            minibatch_states = np.random.choice(self.memory, size=self.minibatch_size, replace=False)
            minibatch_rewards = []
            for p, a, r, s, t in minibatch_states:
                if t:
                    minibatch_rewards.append(r)
                else:
                    future_value = torch.max(self.target_Q(s))

                    future_reward = self.gamma * future_value
                    minibatch_rewards.append(future_reward)
            pred = self.Q(minibatch_states)
            self.opt.zero_grad()
            loss = self.loss(pred, minibatch_rewards)
            self.opt.step()

        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.num_actions+1)
        else:
            q_vals = self.Q(state)
            action = torch.argmax(q_vals)
        self.previous_action = action
        self.previous_state = state
        return action



class DeepQNetwork(nn.Module):

    def __init__(self, num_actions=8):
        super().__init__()

        self.emb = nn.Embedding(MAX_GLYPH, 16)

        # TODO: these should be 3d since we're embedding
        self.conv1 = nn.Conv2d(16, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)

        self.ff1 = nn.Linear(5888, 256)
        self.ff2 = nn.Linear(256, num_actions)

    def forward(self, x):
        print("start", x.shape)
        x = self.emb(x)
        x = self.conv1(x)
        print("conv1", x.shape)
        x = F.relu(x)

        x = self.conv2(x)
        print("conv2", x.shape)
        x = F.relu(x)

        x = F.max_pool2d(x, 3)
        print("pooled", x.shape)
        x = torch.flatten(x, 1)
        print("flattened", x.shape)
        x = self.ff1(x)
        print("ff1", x.shape)
        x = F.relu(x)
        x = self.ff2(x)
        print("ff2", x.shape)

        result = F.log_softmax(x, dim=1)
        print(result.shape)
        return result


if __name__ == "__main__":
    random_data = torch.rand((1, 1, MAP_SIZE[0], MAP_SIZE[1]))
    net = DeepQAgent()
    #result = net(random_data)
    #print(result)
