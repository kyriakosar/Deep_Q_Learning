#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using code structure from
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time
from DQN import DQNAgent
import gym
from helper import LearningCurvePlot, smooth

def average_over_repetitions(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment, n_repetitions, smoothing_window):

    reward_results = np.empty([n_repetitions,num_episodes]) # Result array
    now = time.time()
    env = gym.make('CartPole-v1')
    
    print("Experiment in progress...")
    for rep in range(n_repetitions): # Loop over repetitions
        if use_replay_buffer == True and use_target_network == False:
            print('Replay buffer')
            print("Experiment Repetition: ",rep + 1)
            agent = DQNAgent(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment)
           
            #train the agent
            episode_rewards = agent.train(env)
        elif use_replay_buffer == False and use_target_network == True:
            print('Target network')
            print("Experiment Repetition: ",rep + 1)
            agent = DQNAgent(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment)
            #train the agent
            episode_rewards = agent.train(env)
        elif use_replay_buffer == True and use_target_network == True:
            print('Replay buffer and target network')
            print("Experiment Repetition: ",rep + 1)
            agent = DQNAgent(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment)

            #train the agent
            episode_rewards = agent.train(env)
        elif use_replay_buffer == False and use_target_network == False:
            print('Without Replay buffer or target network')
            print("Experiment Repetition: ",rep + 1)
            agent = DQNAgent(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment)

            #train the agent
            episode_rewards = agent.train(env)

        reward_results[rep] = episode_rewards
        
    # print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    print("the experiment took {} minutes".format((time.time()-now)/60))
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

def experiment():
    
    # Experiment parameters   
    n_repetitions = 15 # number of repetitions
    smoothing_window = 101
   
    
    # Parameters 
    num_episodes = 1000  # number of episodes
    policy = 'annealing_egreedy' # 'egreedy' or 'softmax' or 'annealing_egreedy' or 'novelty'
    model = 'DQN' # 'DQN' or 'DoubleDQN'
    epsilon = 0.9
    plot = False
    experiment = True


    state_dim=4
    action_dim=2 
    hidden_dim=64
    lr=0.001
    gamma=0.9
    buffer_size=10000
    batch_size=64
    target_update=100
    max_steps=500
    eps_start=1.0
    eps_end=0.01
    eps_decay=0.7
    temp=0.2
    novelty=0.5
    tuning=False
    
    


    #Experiment 1:DQN-ER vs DQN-TN vs DQN-ER-TN
    Plot = LearningCurvePlot(title = 'Comparison of DQN with DQN-ER and DQN-TN and DQN-ER-TN')

    use_replay_buffer = True
    use_target_network = True
   
    learning_curve = average_over_repetitions(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment, n_repetitions, smoothing_window)
    Plot.add_curve(learning_curve,label="DQN")

    use_replay_buffer = False
    use_target_network = True

    learning_curve = average_over_repetitions(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment, n_repetitions, smoothing_window)
    Plot.add_curve(learning_curve,label="DQN-ER")


    use_replay_buffer = True
    use_target_network = False
          
    learning_curve = average_over_repetitions(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment, n_repetitions, smoothing_window)
    Plot.add_curve(learning_curve,label="DQN-TN")    


    use_replay_buffer = False
    use_target_network = False
   
    learning_curve = average_over_repetitions(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment, n_repetitions, smoothing_window)
    Plot.add_curve(learning_curve,label="DQN-ER-TN")
    
    Plot.save('plots/experiment_DQN_ER_TN.png')


    #Experiment 2: DQN vs DoubleDQN vs DuelingDQN
    Parameters 
    use_replay_buffer = True
    use_target_network = True
    num_episodes = 1000  # number of episodes
    policy = 'annealing_egreedy' # 'egreedy' or 'softmax' or 'annealing_egreedy' or 'novelty'
    model = 'DQN' 
    target_update = 100 
    plot = False
    experiment = True
    hidden_dim=64
    lr=0.001 
    gamma=0.9 
    eps_decay=0.7 
    batch_size=64
    

    Plot = LearningCurvePlot(title = 'Comparison of DQN and DoubleDQN and DuelingDQN')

    learning_curve = average_over_repetitions(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment, n_repetitions, smoothing_window)
    Plot.add_curve(learning_curve,label="DQN")


    model = 'DoubleDQN' 
    target_update = 100
    hidden_dim=32
    lr=0.001
    gamma=0.9 
    eps_decay=0.9 
    batch_size=32

    learning_curve = average_over_repetitions(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment, n_repetitions, smoothing_window)
    Plot.add_curve(learning_curve,label="DoubleDQN")

    model = 'DuelingDQN'
    target_update = 100
    hidden_dim= 64
    lr=0.001
    gamma=0.7
    eps_decay=0.99 

    learning_curve = average_over_repetitions(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment, n_repetitions, smoothing_window)
    Plot.add_curve(learning_curve,label="DuelingDQN")

    Plot.save('plots/experiments_DQN_models_DQN.png')


    # Experiment 3: Exploration strategies

    Plot = LearningCurvePlot(title = 'Comparison of DQN with different exploration strategies')

    model = 'DQN' 
    use_replay_buffer = True
    use_target_network = True
    epsilon = 0.9
    target_update = 100 
    plot = False
    experiment = True
    hidden_dim=64
    lr=0.001 
    gamma=0.9 
    eps_decay=0.7 
    batch_size=64
    use_replay_buffer = True
    use_target_network = True


    policy = 'annealing_egreedy'

    learning_curve = average_over_repetitions(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment, n_repetitions, smoothing_window)
    Plot.add_curve(learning_curve,label="annealing_egreedy")

    lr = 0.001
    gamma = 0.9
    batch_size = 64
    epsilon = 0.2
    policy = 'egreedy'

    learning_curve = average_over_repetitions(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment, n_repetitions, smoothing_window)
    Plot.add_curve(learning_curve,label="egreedy")

    lr = 0.0001
    gamma = 0.9
    batch_size = 32
    temp = 0.2
    policy = 'softmax'

    learning_curve = average_over_repetitions(state_dim, action_dim, hidden_dim, lr, gamma, buffer_size, batch_size, target_update,
            num_episodes,policy,model,epsilon, max_steps, eps_start, eps_end, eps_decay,temp,
            novelty,plot,tuning, use_replay_buffer, use_target_network,experiment, n_repetitions, smoothing_window)
    Plot.add_curve(learning_curve,label="softmax")


    Plot.save('plots/experiments1_DQN_explorations.png')
   
if __name__ == '__main__':
    experiment()


