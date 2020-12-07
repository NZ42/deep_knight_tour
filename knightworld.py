import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, shape, start):
        self.shape = shape
        self.start = torch.as_tensor(start).detach()
        self.obstacles = []
        self.state = self.start.clone().detach()
        self.state_matrix = self.get_state_matrix()
        #actions ordered clock-wise:     NNE,     NEE,     SEE,    SSE,    SSW,     SWW,     NWW,      NNW
        self.action_list = torch.tensor([[1, -2], [2, -1], [2, 1], [1, 2], [-1, 2], [-2, 1], [-2, -1], [-1, -2]]).detach()

    def get_state(self):
        return self.state

    def get_state_matrix(self):
        matrix_state = torch.zeros(self.shape, device=device).detach()
        matrix_state.T[self.state[0], self.state[1]] = 1
        return matrix_state

    def get_data(self):
        matrix_state = (self.state_matrix == 1).type(torch.FloatTensor)
        matrix_obstacles = (self.state_matrix == -1).type(torch.FloatTensor)
        matrix_to_occupy = (self.state_matrix == 0).type(torch.FloatTensor)
        return torch.stack([matrix_state, matrix_obstacles, matrix_to_occupy])


    def move(self, action):
        old_state = self.get_data()
        self.state_matrix[self.state[0], self.state[1]] = -1
        self.state = self.state + self.action_list[action]
        
        if any(self.state >= torch.as_tensor(self.shape)) or any(self.state < torch.tensor([0, 0])):
            return (old_state, -100, True, None)

        for obstacle in self.obstacles:
            if all(self.state == obstacle):
                return (old_state, -100, True, None)

        self.state_matrix[self.state[0], self.state[1]] = 1
        state = self.get_data()
        assert state.shape == torch.Size([3, 8, 8])
        return (old_state, 1, False, state)
    
    def reset(self):
        self.__init__(self.shape, self.start)
