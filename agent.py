#DQN agent, adapted for Knight Tour from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import torch
import torch.nn as nn
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple
from knightworld import KnightWorld

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
        self.conv1 = nn.Conv2d(1, 8, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 2, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        linear_input_size = conv2d_size_out(conv2d_size_out(self.h)) * conv2d_size_out(conv2d_size_out(self.w)) * 32
        self.head = nn.Linear(linear_input_size, 8)
    
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = nn.functional.relu(x)
        x = self.bn2(self.conv2(x))
        x = nn.functional.relu(x)
        return self.head(x.view(x.shape[0], -1))


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
            return torch.tensor([random.randrange(8)], device=device)
        else:
            state_for_nn = self.state.unsqueeze(0).unsqueeze(0)
            return self.policy_net(state_for_nn).max(1)[1]

    def optimize(self):
        if self.memory.length() < self.batch_size:
            return
            
        transitions = self.memory.sample_batch(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_terminal_next_states = torch.stack([s for s in batch.next_state if s is not None]).unsqueeze_(1)
        
        state_batch = torch.stack(batch.state).unsqueeze_(1)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.cat(batch.reward).unsqueeze_(1)

        state_q_values = self.policy_net(state_batch)
        state_q_values = state_q_values.gather(1, action_batch)
        
        #q value of 0 if state is terminal
        next_state_q_values = torch.zeros(self.batch_size, device=device)
        next_state_q_values[non_terminal_mask] = self.target_net(non_terminal_next_states).max(1)[0].detach()
        next_state_q_values = next_state_q_values.unsqueeze(1)

        expected_state_q_values = (next_state_q_values * self.gamma) + reward_batch

        #using Huber loss as in DQN paper
        loss = torch.nn.functional.smooth_l1_loss(state_q_values, expected_state_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_duration, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated



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
                    self.plot_durations()
                    self.episode_duration.append(t + 1)
                    break
            

            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return self.episode_duration