import torch
from agent import KnightAgent
from knightworld import KnightWorld

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOARD_SHAPE = (8, 8)
KNIGHT_START = (3, 3)
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1
STEPS_BEFORE_EPS_END = 100000
EPS_END = 0.1
REPLAY_MEMORY_SIZE = 10000
TARGET_UPDATE = 100
OPTIMIZER = torch.optim.RMSprop
NUM_EPISODES = 10000

def main():
    knight_world = KnightWorld(BOARD_SHAPE, KNIGHT_START)
    print("Knight world created")
    agent = KnightAgent(knight_world, BATCH_SIZE, GAMMA, TARGET_UPDATE, \
                        EPS_START, STEPS_BEFORE_EPS_END, EPS_END, \
                         REPLAY_MEMORY_SIZE, OPTIMIZER)
    print("Knight created, beginning learning phase")
    agent.learn(NUM_EPISODES)


if  __name__ == "__main__":
    main()