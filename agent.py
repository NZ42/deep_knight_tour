import torch
import torch.nn as nn
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

class DQN(nn.Module):
    def __init__(self, h, w):
        super(DQN, self).__init__()
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
        return self.head(nn.Linear(x.view(x.shape(0), -1)))


class KnightAgent(object):
    def __init__(self, world):
        self.world = world
        self.cumulative_reward = 0
    
    def (self, hidden_layers):


    

