import os
import signal
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


clip_range = 0.2
gamma = 0.99
lam = 0.95
learning_rate = 0.001

hidden_size = 256

HORIZON = 2184
train_iter = 2

save_interval = 5

# True -> train
# False -> inference
is_train = True


class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


class PPO(nn.Module):
    def __init__(self, output_shape):
        super(PPO, self).__init__()
        self.buffer = []
        input_shape = 1 + 20 + 4
        
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.logits_net = nn.Linear(hidden_size, output_shape)
        self.v_net  = nn.Linear(hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.logits_net(x)
        return Categorical(logits=x)
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.v_net(x)
        return v
      
    def store(self, transition):
        self.buffer.append(transition)
        
    def update(self):
        states = torch.tensor([e[0] for e in self.buffer], dtype=torch.float)
        actions = torch.tensor([[e[1]] for e in self.buffer])
        rewards = torch.tensor([[e[2]] for e in self.buffer], dtype=torch.float)
        next_states = torch.tensor([e[3] for e in self.buffer], dtype=torch.float)
        probs = torch.tensor([[e[4]] for e in self.buffer], dtype=torch.float)
        dones = torch.tensor([[1-e[5]] for e in self.buffer])
        self.buffer = []

        for _ in range(train_iter):
            td_target = rewards + gamma * self.v(next_states) * dones
            delta = td_target - self.v(states)
            delta = delta.detach().numpy()

            advantages = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lam * advantage + delta_t[0]
                advantages.append([advantage])
            advantages.reverse()
            advs = torch.tensor(advantages, dtype=torch.float)

            pi = self.pi(states)
            ratio = torch.exp(pi.log_prob(actions) - torch.log(probs))  
            cilp_ratio = torch.clamp(ratio, 1-clip_range, 1+clip_range)

            pi_loss = -torch.mean(torch.min(ratio*advs, cilp_ratio*advs))
            vf_loss = torch.mean(torch.pow(self.v(states) - td_target.detach(), 2))

            loss = pi_loss + vf_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        

def main():
    env = FactoryEnv(is_train)
    killer = GracefulKiller()

    model1 = PPO(10)
    model2 = PPO(4)
    if os.path.exists("save.pt"):
        print("model loaded!")
        checkpoint = torch.load("save.pt")
        model1.load_state_dict(checkpoint["model1"])
        model2.load_state_dict(checkpoint["model2"])

    if not is_train:
        model1.eval()
        model2.eval()

    for i in itertools.count():
        s = env.reset()
        done = False
        while not done:
            for t in range(HORIZON):
                def get_action(model):
                    pi = model.pi(torch.from_numpy(s).float())
#                     if is_train:
                    a = pi.sample()
#                     else:
#                         a = torch.argmax(pi.probs)
                    return a.item(), pi.probs[a].item()
                
                a1, prob1 = get_action(model1)
                a2, prob2 = get_action(model2)
                
                if t % 126 == 0:
                    env.step1(a1)
                next_s, r, done, info = env.step2(a2)
                if t % 126 == 0:
                    model1.store((s, a1, r, next_s, prob1, done))
                model2.store((s, a2, r, next_s, prob2, done))

                s = next_s

                if done:
                    break

            model1.update()
            model2.update()

        if not is_train:
            env.save_csv()
            break

        if i%save_interval==0 and i!=0:
            torch.save({"model1": model1.state_dict(),
                        "model2": model2.state_dict()}, f"save_{i}.pt")
            if killer.kill_now:
                if input('Terminate training (y/[n])? ') == 'y':
                    env.save_csv()
                    break
                killer.kill_now = False


if __name__ == '__main__':
    main()