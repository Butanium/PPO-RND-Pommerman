# -*- coding: utf-8 -*-


# Load previous created results
# Set to False if:
#     * You have installed Pommerman, and
#     * You have installed docker, and
#     * You want to wait for the games to be replayed
#     * Don't worry we will not train again, we will just load a pre-trained network
loadPrevious = True

"""## Import libraries"""
import pommerman
from pommerman import agents
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Normal

# Our own files
from convertInputMapToTrainingLayers import *

"""# Setting up the network

First our main network, an ActorCritic network
"""

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic_con = nn.Sequential(
            nn.Conv2d(in_channels=7,
                      out_channels=64,
                      kernel_size=3,
                      padding=0),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=0),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=0),
            nn.ReLU()
        )
        self.critic_linear = nn.Sequential(
            nn.Linear(3*3*64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor_con = nn.Sequential(
            nn.Conv2d(in_channels=7,
                      out_channels=64,
                      kernel_size=3,
                      padding=0),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=0),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=0),
            nn.ReLU()
        )
        self.actor_linear = nn.Sequential(
            nn.Linear(3*3*64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )

        self.log_std = nn.Parameter(torch.ones(num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic_con(x)
        value = self.critic_linear(value.view(-1, 3*3*64))

        mu    = self.actor_con(x)
        mu    = self.actor_linear(mu.view(-1, 3*3*64))

        std1  = self.log_std.exp()
        std   = std1.expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
"""Hyper parameters:"""

num_inputs       = 324
num_outputs      = 6
hidden_size      = 1024
lr               = 1e-6
lr_RND           = 1e-3
mini_batch_size  = 5
ppo_epochs       = 4
max_frames       = 1500000
frame_idx        = 0
game_idx         = 0
device           = "cpu" # Hard-coded since we have a GPU, but does not want to use
clip_param       = 0.2

"""# Training Networks (loading pre-trained network)

We trained 4 identical networks, with random start weights. After 1 500 000 frames of training each we took the two best networks, and trained a copy of them with a renewed reward function, until 3 200 000 Frames each.

The win rate is based on test play against 3 simple agents in FFA mode:

![01](images/TrainAI01.png "AI 01")
![02](images/TrainAI02.png "AI 02")
![03](images/TrainAI03.png "AI 03")
![04](images/TrainAI04.png "AI 04")
"""

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
model = torch.load("models/newAI02_from_oldAI04.pth", map_location=device)
# Correctness warning may appear, do not worry, this is normal

"""# Results


Pre-computed win rates:

| # Game | # Network controlled AI | # Simple Agents | # Random Agents | Win Rate |
|--------|-------------------------|-----------------|-----------------|----------|
|    500 |                       1 |               0 |               3 |    100 % |
|    300 |                       1 |               1 |               2 |     85 % |
|    300 |                       1 |               2 |               1 |     64 % |
|    300 |                       1 |               3 |               0 |     53 % |
|    300 |                       2 |               1 |               1 |     96 % |
|    300 |                       2 |               2 |               0 |     81 % |

### Setup environment for calculating results
"""


def playGame(env):
    state = env.reset()
    done = False
    vs = []
    rs = [0]
    while not done:
        stateOrginal = state
        state = torch.FloatTensor(stateToTorch(state)).to(device)
        dist, v = model(state)
        vs.append(float(v))
        actionsList = env.act(stateOrginal)
        state, reward, done, info = env.step([dist.mean.cpu().data.numpy()[0].argmax()] + actionsList[1:])
        rs.append(reward[0] + rs[-1])
        print(reward)
    if "winners" in info:
        if 0 in info["winners"]:
            plt.plot(vs)
            plt.plot(rs[1:])
            plt.show()
            return "Won"
            
        else:
            return "Lost"
    else:
        return "Tie"

# Create a set of agents (exactly four)
agent_list = [
    agents.RandomAgent(), # Does not matter, we control this agent
    agents.SimpleAgent(), # Replace with RandomAgent for easier games
    agents.RandomAgent(), # Replace with RandomAgent for easier games
    agents.RandomAgent(), # Replace with RandomAgent for easier games
]
# Make the "Free-For-All" environment using the agent list
env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)

gamesPlayed    = 0
gamesWon       = 0
gamesThatCount = 0 # Games that end in a tie is replayed, and therefore
for _ in range(100):
    result = playGame(env)
    gamesPlayed += 1
    if result == "Won":
        gamesWon += 1
        break
    if result != "Tie":
        gamesThatCount += 1
    if gamesThatCount != 0:
        print("Played " + str(gamesPlayed) + " games, with a win rate of " + str(gamesWon/float(gamesThatCount) * 100.0) + "%  (with " + str(gamesThatCount) + " games not ending in a tie)")

"""#### Please note this make take upwards of 300 games to stabilize"""