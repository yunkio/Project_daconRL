import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class DQNAgent():
    def __init__(self, state_size, action_size, frame_size=1, state_dict=None, target_update_interval=50, train=True):
        self.state_size = state_size
        self.action_size = action_size
        self.target_update_interval = target_update_interval
        self.episode = 0

        self.discount_factor = 0.95
        self.learning_rate = 0.000005
        self.eps = 1.0 if train else 0.00001
        self.eps_decay_rate = 0.999
        self.eps_min = 0.05
        self.batch_size = 64

        self.memory = deque(maxlen=10000)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
            self.target_model.load_state_dict(state_dict)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()

    def get_action(self, x):
        if self.eps_min < self.eps:
            self.eps *= self.eps_decay_rate
        if np.random.rand() <= self.eps:
            return random.choice(np.arange(self.action_size))
        else:
            x = self.preprocess_state(x)
            x = torch.FloatTensor(x)
            return int(torch.argmax(self.model(x.view(1, self.state_size))))

    def preprocess_state(self, state):
        # new_state = state[[0, 1, 3, 4, 6, 7, 9, 10], :]
        return state
    
    def append_sample(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(self.preprocess_state(state)).view(self.state_size)
        next_state = torch.FloatTensor(self.preprocess_state(next_state)).view(self.state_size)
        self.memory.append((state, action, reward, next_state, done))
        if done:
            self.episode += 1
    
    def train_model(self):
        if len(self.memory) < 2000:
            return
        mini_batch = random.sample(self.memory, self.batch_size)

        states = torch.zeros((self.batch_size, self.state_size))
        next_states = torch.zeros((self.batch_size, self.state_size))
        actions = torch.zeros(self.batch_size).type(torch.LongTensor)
        rewards = torch.zeros(self.batch_size)
        dones = torch.zeros(self.batch_size).type(torch.LongTensor)

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions[i] = int(mini_batch[i][1])
            rewards[i] = float(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones[i] = mini_batch[i][4]

        next_q_val = self.target_model(next_states)
        q_val = self.model(states)
        exp_q_val = torch.zeros_like(q_val)

        terminal = torch.where(dones != 0)[0]
        not_terminal = torch.where(dones == 0)[0]

        if terminal.numel() > 0:
            exp_q_val[terminal, actions[terminal]] = rewards[terminal]
        if not_terminal.numel() > 0:
            exp_q_val[not_terminal, actions[not_terminal]] = rewards[not_terminal] + \
                self.discount_factor * torch.max(next_q_val[not_terminal, :], axis=1).values

        loss = self.loss(q_val, exp_q_val)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.episode % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())


        
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layer = layer = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.layer(x)