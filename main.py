import torch
import matplotlib
import matplotlib.pyplot as plt
from agent import KnightAgent
from knightworld import KnightWorld

device = "cpu"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOARD_SHAPE = (8, 8)
KNIGHT_START = (1, 2)
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
STEPS_BEFORE_EPS_END = 100000
EPS_END = 0.1
REPLAY_MEMORY_SIZE = 10000
TARGET_UPDATE = 100
OPTIMIZER = torch.optim.RMSprop
NUM_EPISODES = 1000

def main():
    knight_world = KnightWorld(BOARD_SHAPE, KNIGHT_START)
    print("Knight world created")
    agent = KnightAgent(knight_world, BATCH_SIZE, GAMMA, TARGET_UPDATE, \
                        EPS_START, STEPS_BEFORE_EPS_END, EPS_END, \
                         REPLAY_MEMORY_SIZE, OPTIMIZER)
    print("Knight created, beginning learning phase")
    episode_durations = agent.learn(NUM_EPISODES)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations)
    plt.show()


if  __name__ == "__main__":
    main()