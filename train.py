from DQN import DQNAgent
import gym
import argparse

""" 
Command line arguments:

python train.py --model DQN or DoubleDQN or DuelingDQN to run the respective model
python train.py --without_ER to run without replay buffer
python train.py --without_TN to run without target network
python train.py --without_ER --without_TN to run without replay buffer and target network
    
"""


def main():

    #parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store", type=str,nargs='?', default='DQN')
    parser.add_argument("--without_ER", action="store_true")
    parser.add_argument("--without_TN", action="store_true")


    args = parser.parse_args()

    env = gym.make('CartPole-v1')

    #set the default values
    use_replay_buffer = True
    use_target_network = True

    #check if the user wants to run without replay buffer or target network
    if args.without_ER and not args.without_TN:
        print("Running without Replay buffer")
        use_replay_buffer = False
    elif args.without_TN and not args.without_ER:
        print("Running without Target network")
        use_target_network = False
    elif args.without_ER and args.without_TN:
        print("Running without Replay buffer and Target network")
        use_replay_buffer = False
        use_target_network = False

    #check which model is selected
    if args.model == 'DQN':
        print('Running with DQN')
    elif args.model == 'DoubleDQN':
        print('Running with DoubleDQN')
    elif args.model == 'DuelingDQN':
        print('Running with DuelingDQN')
        

    #initialize the agent
    agent = DQNAgent(state_dim=4, action_dim=2, hidden_dim=64, lr=0.001, gamma=0.9, buffer_size=10000, batch_size=64, target_update=100,
    num_episodes=1000,policy='annealing_egreedy',model = args.model ,epsilon=0.9, max_steps=500, eps_start=1.0, eps_end=0.01, eps_decay=0.9,temp=0.5,novelty=0.5,plot=True,tuning=False, use_replay_buffer = use_replay_buffer, use_target_network = use_target_network)

    #train the agent
    episode_rewards = agent.train(env)

if __name__ == "__main__":
    main()