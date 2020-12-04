import torch

#actions, ordered clockwise
NNE = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
NEE = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
SEE = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
SSE = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.bool)
SSW = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.bool)
SWW = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.bool)
NWW = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool)
NNW = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool)

class KnightWorld(object):
    def __init___(self, shape, start):
        self.shape = shape
        self.start = start
        self.obstacles = torch.tensor([])
        self.state = self.start
        #actions ordered clock-wise: NNE, NEE, SEE, SSE, SSW, SWW, NWW, NNW
        self.action_list = torch.tensor([[1, -2], [2, -1], [2, 1], [1, 2], [-1, 2], [-2, 1], [-2, -1], [-1, -2]])

    def get_state(self):
        return self.state

    def get_state_matrix(self):
        matrix = torch.zeros(self.shape, dtype=torch.int8)
        matrix[self.state] = 1
        for obstacle in self.obstacles:
            matrix[obstacle] = -1
        return matrix

    def move(self, action):
        old_state = self.state
        self.state = self.state + self.action_list[action]

        if any(self.state > self.shape) or any(self.state < torch.tensor([0, 0])):
            return (old_state, 0, True, None)

        for obstacle in self.obstacles:
            if self.state == obstacle:
                return (old_state, 0, True, None)
        
        return (old_state, 1, False, self.state)
    
    def reset(self):
        self.state = self.start
        self.obstacles = torch.tensor([])
