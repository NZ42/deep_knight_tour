#DQN agent, adapted for Knight Tour from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import torch
import torch.nn as nn
import random
from itertools import count
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def length(self):
        return len(self.memory)

    def push_transition(self, *args):
        if self.length() < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch

class KnightDQN(nn.Module):
    def __init__(self, h, w):
        super(KnightDQN, self).__init__()
        self.h = h
        self.w = w
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        def conv2d_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        linear_input_size = conv2d_size_out(self.h) * conv2d_size_out(self.w) * 8
        self.head = nn.Linear(linear_input_size, 8)
    
    def forward(self, x):
        x = nn.ReLU(self.bn1(self.conv1(x)))
        return self.head(x.view(x.shape(0), -1))


class KnightAgent(object):
    def __init__(self, world, batch_size, gamma, target_update, eps_start, steps_before_end, eps_end, memory_size, optim):
        self.world = world
        self.state = self.world.get_state_matrix()
        self.batch_size = batch_size
        self.gamma = gamma
        self.steps_done = 0
        self.eps_start = eps_start
        self.steps_before_end = steps_before_end
        self.eps_end = eps_end
        self.target_update = target_update
        self.policy_net = KnightDQN(self.world.shape[0], self.world.shape[1]).to(device)
        self.target_net = KnightDQN(self.world.shape[0], self.world.shape[1]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim(self.policy_net.parameters())
        self.memory = ReplayMemory(memory_size)
        self.episode_duration = []

    def choose_action(self):
        #epsilon greedy with epsilon annealed linaerly from self.eps_start to self.eps_end over self.steps_before_end, then self.eps_end
        #linear annealing as in original DQN paper
        sample = random.random()
        if self.steps_done < self.steps_before_end:
            eps = self.eps_start - self.steps_done * (self.eps_start - self.eps_end) / self.steps_before_end 
        else:
            eps = self.eps_end
        
        self.steps_done += 1

        if sample < eps:
            return torch.tensor([[random.randrange(8)]], device=device, dtype=torch.long)
        else:
            return self.policy_net(self.state)

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample_batch(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_terminal_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        #q value of 0 if state is terminal
        next_state_q_values = torch.zeros(self.batch_size, device=device)
        next_state_q_values[non_terminal_mask] = self.target_net(non_terminal_next_states).max(1)[0].detach()

        expected_state_q_values = (next_state_q_values * self.gamma) + reward_batch

        #using Huber loss as in DQN paper
        loss = torch.nn.SmoothL1Loss(state_q_values, expected_state_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()

    def learn(self, n_episodes):
        for episode in range(n_episodes):
            self.world.reset()
            self.state = self.world.get_state_matrix()
            for t in count():
                action = self.choose_action()
                old_state, reward, terminal, self.state = self.world.move(action.item())
                reward = torch.tensor([reward], device=device)

                self.memory.push_transition(old_state, action, self.state, reward)

                self.optimize()

                if terminal:
                    self.episode_duration.append(t + 1)
                    break
        
        if episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        print(f"Training complete, last episode duration: {self.episode_duration[-1]}")
        return self.episode_duration


    

