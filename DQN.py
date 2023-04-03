import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
from tqdm import tqdm
from model_architecture import DQN, DuelingDQN
import matplotlib.pyplot as plt
import os




# Define the DQN agent
class DQNAgent():
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64, lr=0.001, gamma=0.99, buffer_size=10000, batch_size=32, target_update=100,
    num_episodes=1000,policy='egreedy',model='DQN',epsilon=1.0, max_steps=500, eps_start=1.0, eps_end=0.01, eps_decay=0.995,temp=0.2,novelty=0.5,plot=True, tuning=False,use_replay_buffer=True, use_target_network=True,experiment=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.policy = policy
        self.model = model
        self.tuning = tuning
        self.use_replay_buffer = use_replay_buffer
        self.use_target_network = use_target_network
        self.batch = []
        self.experiment = experiment

        #check if gpu is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #initialize the networks and optimizer based on the selected model
        if self.model == 'DQN' or self.model == 'DoubleDQN':
            self.q_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
            if self.use_target_network:
                self.target_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.target_network.eval()
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        elif self.model == 'DuelingDQN':
            self.q_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
            if self.use_target_network:
                self.target_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.target_network.eval()
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
    
        #initialize the replay buffer if replay buffer is selected
        if self.use_replay_buffer:
            self.replay_buffer = deque(maxlen=self.buffer_size)
     
        
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.temp = temp
        self.novelty = novelty
        self.plot = plot
        
        
        self.steps = 0

    #exploration strategies    
    def act(self, state):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        eps = self.epsilon

        #select action based on the selected policy
        if self.policy == "egreedy" or self.policy == 'annealing_egreedy':
            
            if self.policy == 'annealing_egreedy': 
                eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.eps_decay * self.steps)

            if np.random.rand() < eps:
                action = np.random.randint(self.action_dim)
            else:
                action = torch.argmax(q_values).item()        
        elif self.policy == "softmax":
            action = torch.softmax(q_values / self.temp, dim=-1).data.numpy().squeeze()
            action = np.random.choice(self.action_dim, p=action) 
            
        return action
        
    def update(self):
        #if the replay buffer still not filled, return
        if self.use_replay_buffer and len(self.replay_buffer) < self.batch_size:
            return
        
        #if the replay buffer is not used, batch is used instead
        if self.use_replay_buffer:
            batch = random.sample(self.replay_buffer, self.batch_size)
        else:
            batch = self.batch 

        #extract the data from the batch
        state, action, reward, next_state, done = zip(*batch)
        
        #convert the data to tensor
        state = torch.FloatTensor(np.float32(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        #calculate the q values
        q_values = self.q_network(state).gather(1, action)

        #calculate next q values based on the selected model and if target network is used
        if self.use_target_network and self.model == 'DQN':
            next_q_values = self.target_network(next_state).max(1)[0].unsqueeze(1)
        elif self.use_target_network and self.model == 'DoubleDQN':
            next_actions = self.q_network(next_state).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_state).gather(1, next_actions)
        else:
            next_q_values = self.q_network(next_state).max(1)[0].unsqueeze(1)

        #calculate the expected q values
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)
        
        #calculate the loss
        loss = nn.functional.smooth_l1_loss(q_values, expected_q_values.detach())
        
        #update the main network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #update the target network
        self.steps += 1
        if self.steps % self.target_update == 0 and self.use_target_network:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
    def memorize(self, state, action, reward, next_state, done):
        #if replay buffer is used, append the data to the replay buffer
        if self.use_replay_buffer:
            self.replay_buffer.append((state, action, reward, next_state, done))
        else:
            self.batch = [(state, action, reward, next_state, done)]

    def train(self, env):
        #initialize the list to store the rewards
        episode_rewards = []
        if self.tuning:
                print("Tuning hyperparameters is in progress...")
        #loop through the episodes
        for i in tqdm(range(self.num_episodes)):
            state = env.reset()
            # eps = self.eps_start - i * (self.eps_start - self.eps_end) / (self.num_episodes - 1)
            episode_reward = 0
            #loop through the steps in each episode
            for j in range(self.max_steps):
                #select action based on the selected policy
                action = self.act(state)
                #take the action and get the next state, reward, and done
                next_state, reward, done, _ = env.step(action)
                
                #call the memorize function to store the data in the replay buffer
                self.memorize(state, action, reward, next_state, done)

                #call the update function to update the network
                self.update()
                
                #update the state
                state = next_state
                episode_reward += reward

                #if the episode is done, break the loop
                if done:
                    break
    
            #append the episode reward to the list
            episode_rewards.append(episode_reward)  
            
        #print the average reward  
        if not self.tuning and not self.experiment:
            print("Training is finished. Average Reward = {}".format(np.mean(episode_rewards)))

        #plot the learning curve
        if self.plot:
            plt.plot(episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title("Learning Curve")
            
            try:     
                #save plot
                plt.savefig('plots/learning_curve.png')
                print("Successfully saved plot")
            except:
                print("Error saving plot")

        return episode_rewards


    

