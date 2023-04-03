import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import torch.nn.functional as F


# Define the DQN network architecture
class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Define Dueling DQN network architecture
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DuelingDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
            
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
            
        self.advantage = nn.Linear(self.hidden_dim, self.action_dim)
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        advantage = self.advantage(x)
        value = self.value(x)
        Q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            
        return Q_values

