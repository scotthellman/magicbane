#https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F
from .datastructures import CircularBuffer
import numpy as np

from torch.utils.tensorboard import SummaryWriter



# FIXME: this is just a bandaid
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



class DeepQAgent:

    def __init__(self, num_actions, map_size, vocab_size, max_epsilon=0.95, min_epsilon=0.1,
                 epsilon_decay_window=20000, gamma=0.99, C=1000, memory_size=2000, lr=0.0001,
                 minibatch_size=8, kernel_size=5):
        self.num_actions = num_actions
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_step = (max_epsilon - min_epsilon) / 20
        self.epsilon_reduction_frequency = epsilon_decay_window/20
        self.gamma = gamma
        self.C = C
        self.memory = CircularBuffer(memory_size)
        self.Q = DeepQNetwork(num_actions, map_size, vocab_size, kernel_size=kernel_size)
        self.target_Q = DeepQNetwork(num_actions, map_size, vocab_size, kernel_size=kernel_size)
        self.minibatch_size = minibatch_size
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.reset()

    def build_state(self, obs):
        glyphs = obs["glyphs"]
        return torch.from_numpy(glyphs).int()

    def reset(self):
        self.previous_state = None
        self.previous_action = None
        self.steps = 0
        self.update_target_Q()
        self.total_loss = 0
        self.total_reward = 0

    def update_target_Q(self):
        self.target_Q.load_state_dict(self.Q.state_dict())


    def step(self, state, reward, terminal_state):
        # TODO: handle unfull memory
        state = state.unsqueeze(0)
        self.steps += 1
        if self.steps % self.epsilon_reduction_frequency == 0 and self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_step
            print("Reducing epsilon to", self.epsilon)
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon
        if self.steps % self.C == self.C - 1:
            self.update_target_Q()
            print(f"Total loss {self.total_loss}, total reward {self.total_reward}")
            self.total_loss = 0
            self.total_reward = 0
        if self.previous_state is not None:
            self.memory.insert((self.previous_state, self.previous_action, reward, state, terminal_state))
            #writer.add_scalar("Reward", reward, self.steps)
            self.total_reward += reward
        if len(self.memory) > 2*self.minibatch_size:

            # numpy doesn't like indexing into self.memory directly for some reason
            minibatch_indices = np.random.choice(range(len(self.memory)), size=self.minibatch_size, replace=False)
            minibatch_states = [self.memory[i] for i in minibatch_indices]
            minibatch_y = []
            minibatch_X = []
            minibatch_action_mask = torch.zeros([self.minibatch_size, self.num_actions], dtype=torch.bool)
            for m_idx, (p, a, r, s, t) in enumerate(minibatch_states):
                minibatch_X.append(p)
                minibatch_action_mask[m_idx, a] = True
                if t:
                    minibatch_y.append(r)
                else:
                    future_value = torch.max(self.target_Q(s))

                    future_reward = self.gamma * future_value
                    minibatch_y.append(future_reward)
            minibatch_y = torch.FloatTensor(minibatch_y)
            full_preds = self.Q(torch.cat(minibatch_X))
            target_action_preds = torch.masked_select(full_preds, minibatch_action_mask)
            self.opt.zero_grad()
            loss = self.loss(target_action_preds, minibatch_y)
            self.opt.step()
            self.total_loss += loss
            #writer.add_scalar("Loss", loss, self.steps)

        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            q_vals = self.Q(state)
            action = torch.argmax(q_vals).item()
        self.previous_action = action
        self.previous_state = state
        return action


class DeepQNetwork(nn.Module):

    def __init__(self, num_actions, map_size, vocab_size, kernel_size=5, emb_size=16):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_size)

        first_channels = 32
        final_channels = 32

        # TODO: should these be 3d since we're embedding?
        self.conv1 = nn.Conv2d(emb_size, first_channels, kernel_size, 1)
        self.conv2 = nn.Conv2d(first_channels, final_channels, kernel_size, 1)


        final_height = (map_size[0] - 2*(kernel_size-1))//3
        final_width = (map_size[1] - 2*(kernel_size-1))//3
        conv_neurons = final_channels * final_height * final_width

        self.ff1 = nn.Linear(conv_neurons, 256)
        self.ff2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = self.emb(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 3)
        x = torch.flatten(x, 1)
        x = self.ff1(x)
        x = F.relu(x)
        x = self.ff2(x)

        result = F.log_softmax(x, dim=1)
        return result


if __name__ == "__main__":
    random_data = torch.rand((1, 1, MAP_SIZE[0], MAP_SIZE[1]))
    net = DeepQAgent()
    #result = net(random_data)
    #print(result)
