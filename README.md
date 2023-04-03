# Assignment: Deep Q Learning
# Course: Reinforcement Learning, Leiden University
# Authors: Kyriakos Aristidou, Antonia Christooulou, Georgios Doukeris 
# April 2023

This file contains the instructions of the solution of the 2nd assignment of the course Reinforcement Learning in Leiden University. The assignment concerns the implementation of a Deep Q Learning algorithm for the CartPole V1 challenge from OpenAI. To do that, we use reinforcement learning algorithms that are able to address problems with continuous state-action spaces.


## Create a virtual environment and use the requirements file to run the code successfully
 
```
 pip install -r requirements.txt

```

# Create a folder in the main directory of the project: "plots" to save the plots of the experiments

```
mkdir plots

```

# Run the experiments

You can run the experiments by running the following command in the main directory : `python experiment.py`

The plots of the experiments will be saved in the folder "plots" in the main directory of the project.



# Run to train with different models

You can run the training of the model by running the following command in the main directory : `python train.py`

This by default will train using the "DQN" model.


If you want to train the model with the other models, you can run the following commands:

When you want to train the model with the DoubleDQN model, run the following command: `python train.py --model DoubleDQN`
When you want to train the model with the DuelingDQN model, run the following command: `python train.py --model DuelingDQN`


# Run to train the DQN model with removed parts

You can run the training of the model with removed parts by running the following commands in the main directory :

When you want to train the model without Replay Buffer, run the following command: `python train.py --without_ER`
When you want to train the model without Target Network, run the following command: `python train.py --without_TN`
When you want to train the model without both Replay Buffer and Target Network, run the following command: `python train.py --without_ER --without_TN`


# Model parameters:

The parameters of the model can be changed in the file "train.py" in the main directory of the project there that the agent is initilized. The default parameters are the following:

state_dim=4,
action_dim=2, 
hidden_dim=64, 
lr=0.001, 
gamma=0.9, 
buffer_size=10000,
batch_size=64,
target_update=100,
num_episodes=1000,
policy='annealing_egreedy', [annealing_egreedy, egreedy, softmax]
epsilon=0.9, 
max_steps=500, 
eps_start=1.0, 
eps_end=0.01, 
eps_decay=0.9,
temp=0.5,
novelty=0.5,
plot=True,

