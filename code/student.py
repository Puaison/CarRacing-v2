import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(2096442)
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt

class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        print(self.device)
        print("student")
        self.skip_frames=64 #frames skipped at the beginning of each episode
        self.stacked_frames=4 
        self.count_frames=0 #counter of frame for each episode
        self.last_frames =np.zeros((self.stacked_frames, *(84,84)), dtype=np.float32) #array containing the last 4 frames
        self.network = Q_Net().to(self.device)
        self.initialize()
        self.step_count = 0
        self.episode = 0

    def act(self, state):

        if self.count_frames<self.stacked_frames: #STACKING THE FIRST 4 FRAMES FOR NEW STATE
            s=self.preprocess(state)
            #j=self.count_frames%(self.stacked_frames)
            self.last_frames[self.count_frames]=s
            self.count_frames+=1
            return 0

        ##A REGIME
        s=self.preprocess(state)

        self.last_frames[:-1] = self.last_frames[1:]  # moves all the 4 frames to the left in order to save the new frame and remove the oldest one
        self.last_frames[-1] = s # saving the last frame
        a=self.network.greedy_action(torch.as_tensor(self.last_frames).to(self.device))
        return a

    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.rewards = 0
        self.step_count = 0
        self.episode = 0
    
    #function that resets the environment and skips the first 64 frames
    def super_restart(self,env):
        self.count_frames=0
        _,_=env.reset()
        for i in range(self.skip_frames):
            if self.count_frames<self.skip_frames - self.stacked_frames: #skipping the first 60 frames
                s, r, terminated, truncated, info = env.step(0)
                self.count_frames+=1
                continue

            if self.count_frames<self.skip_frames: #stacking the last 4 frames before starting 
                s, r, terminated, truncated, info = env.step(0)
                s=self.preprocess(s)
                j=self.count_frames%(self.skip_frames - self.stacked_frames)
                self.last_frames[j]=deepcopy(s)
                self.count_frames+=1
                continue

            print("fuori dal ciclo")
        #self.env=env
        self.s_0=deepcopy(self.last_frames)
        return env

    #function for cropping the image to 84x84 in grayscale and then normalizing 
    def preprocess(self,image,normalization=True):
        #image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=image[:84, 6:90]
        if normalization:
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.0
        else:
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def take_step(self, mode='exploit'):
        # choose action with epsilon greedy
        if mode == 'explore':
                action = self.env.action_space.sample()
        else:
                action = self.network.greedy_action(torch.as_tensor(self.s_0).to(self.device))
        patience_truncation=False #truncation due to the trigger of max patience
        #simulate action
        s_1, r, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated #when the tiles are all visited or timeout
        s_1=self.preprocess(s_1)
        self.last_frames[:-1] = self.last_frames[1:]  # moves all the 4 frames to the left in order to save the new frame and remove the oldest one
        self.last_frames[-1] = deepcopy(s_1) #saving the last frame

        self.step_count += 1
        if r >=0: #patience mechanism when the agent is not collecting positive rewards
            self.actual_patience = 0
        else:
            self.actual_patience+= 1
            if self.actual_patience == self.max_patience: #trigger of max patience 
                patience_truncation = True
                self.actual_patience = 0
                r=-100
        #put experience in the buffer
        self.buffer.add(deepcopy(self.s_0), action, r, int(done), deepcopy(self.last_frames)) #we save only done, not patience_truncation

        self.rewards += r

        self.s_0 = deepcopy(self.last_frames)

        if done or patience_truncation:
            self.env=self.super_restart(self.env) #prepare the environment for new episode
            self.actual_patience = 0
            done=True
        return done

    # Implement DDQN training algorithm
    def train(self):
        self.env= gym.make('CarRacing-v2', continuous=self.continuous,max_episode_steps=930)#, render_mode='human')
        self.gamma = 0.99
        self.max_patience=100
        self.actual_patience=0

        #Variables for managing epsilon-greedy algorithm
        self.epsilon = 0.7 
        self.min_epsilon=0.05
        self.epsilon_decay=0.99
        
        self.learning_rate=1e-4
        self.target_network = deepcopy(self.network).to(self.device)
        self.network.set_learning_rate(self.learning_rate)
        self.network_update_frequency=10 #every 10 frames sampling from buffer for updating normal network
        self.network_sync_frequency=100 #every 100 frames copying the normal network to the target one
    
        self.window = 50
        self.best_mean_reward=-1000
        self.max_episodes=3000
        self.reward_threshold =600

        self.batch_size = 64
        self.exponent_alpha=0.6 #aplha of the stochastic prioritization
        self.exponent_beta_initial=0.4 #initial beta for IS calculation
        self.exponent_beta_final=1 #final beta for IS calculation
        self.buffer=PrioritizedExperienceReplayBuffer(alpha=self.exponent_alpha, beta=self.exponent_beta_initial,burn_in=20000, buffer_size=100000,device=self.device)
        self.loss_function = nn.MSELoss()
        self.env=self.super_restart(self.env)
        # Populate replay buffer
        print("Populating replay buffer")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')
        print("Burn in finished")
        ep = 0
        training = True
        while training:
            self.rewards = 0
            done = False
            #computing gradually new beta (IS) for each episode 
            actual_beta=self.exponent_beta_initial + (self.exponent_beta_final - self.exponent_beta_initial) * (ep / (self.max_episodes - 1))
            print(actual_beta)
            self.buffer.change_beta(actual_beta)
            while not done:
                p = np.random.random()
                if p < self.epsilon:#epsilon-greedy
                    done = self.take_step(mode='explore')
                    # print("explore")
                else:
                    done = self.take_step(mode='exploit')
                    # print("train")
                # Update network
                if self.step_count % self.network_update_frequency == 0:
                    self.update()
                # Sync networks

                if self.step_count % self.network_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

                if done:
                    if self.epsilon >= self.min_epsilon:
                        self.epsilon = self.epsilon * self.epsilon_decay
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    if len(self.update_loss) == 0:
                        self.training_loss.append(0)
                    else:
                        self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(self.training_rewards[-self.window:])
                    mean_loss = np.mean(self.training_loss[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    if mean_rewards > self.best_mean_reward: 
                        self.best_mean_reward = mean_rewards
                        self.intermediate_save()
                        print("Saved model at ep:", ep, " with mean reward:", mean_rewards)
                    print(
                        "\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}   mean loss = {:.2f}\t\t".format(
                            ep, mean_rewards, self.rewards, mean_loss), end="")

                    if ep >= self.max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        #break
        # plot
        self.plot_training_rewards()
        self.save()

    #train the normal network with samples from the buffer
    def update(self):
        self.network.optimizer.zero_grad()
        batch, idxs,weights = self.buffer.sample(batch_size=self.batch_size) # sample from the buffer with PER mechanism
        loss, td_error= self.calculate_loss(batch,weights=weights) #DDQN Update
        self.buffer.update_priorities(idxs, td_error.cpu().numpy()) #update the priorities in the buffer for sampled transition

    def calculate_loss(self, batch, weights=None):
        #extract info from batch
        states, actions, rewards, dones, next_states = batch[0],batch[1],batch[2],batch[3],batch[4]
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        ###############
        # DDQN Update #
        ###############
        qvals = self.network.get_qvals(states) #this is Q(s-1) of dim [num_batch,num_actions] 
        qvals = torch.gather(qvals, 1, actions) #this is Q(s-1,a-1) of dim [num_batch,1]
        next_qvals=self.network.get_qvals(next_states)#this is Q(s)
        target_next_qvals= self.target_network.get_qvals(next_states)#this is Q_target(s)
        max_indices=torch.max(next_qvals, dim=-1)[1] #we get the action that makes Q(s) maximum
        selected_target_values = target_next_qvals[torch.arange(target_next_qvals.size(0)), max_indices].reshape(-1, 1)#we substitute the actions calculated in the line before in Q_target(s,a')
        target_qvals = rewards + (1 - dones)*self.gamma*selected_target_values

        assert qvals.shape == target_qvals.shape, f"{qvals.shape}, {target_qvals.shape}"
        td_error = torch.abs(qvals - target_qvals).detach()
        loss = torch.mean((qvals - target_qvals)**2 * weights)
        loss.backward()
        self.network.optimizer.step()

        self.update_loss.append(loss.item())
        return loss, td_error
    
    def plot_training_rewards(self):
        #save_path = '/content/drive/My Drive/Reinforcement_Learning_HW/'
        #os.makedirs(save_path, exist_ok=True)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.mean_training_rewards, label='Mean Rewards', color='orange')
        plt.title('Training and Mean Rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episodes')
        plt.legend()
        plt.savefig('mean_training_rewards.png')
        plt.show()
        plt.clf()


    def forward(self, x):
        # TODO
        return x

    def save(self):
        #save_path = '/content/drive/My Drive/Reinforcement_Learning_HW/'
        #os.makedirs(save_path, exist_ok=True)
        torch.save(self.network.q_net.state_dict(),'model_final.pt')

    def intermediate_save(self):
        torch.save(self.network.q_net.state_dict(),'model_intermediate.pt')

    def load(self):
        #save_path = '/content/drive/My Drive/Reinforcement_Learning_HW/'
        #os.makedirs(save_path, exist_ok=True)
        self.network.q_net.load_state_dict(torch.load('model_fast.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

#DQN architecture used by DeepMind in Atari Games
class DQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=5, activation=F.relu):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

#Controller of the network
class Q_Net(nn.Module):
    def __init__(self, lr = 1e-4, dqn = True):
        super(Q_Net, self).__init__()
        self.q_net = DQN()
        print("Q_network: ", self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def set_learning_rate(self,lr):
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def get_qvals(self, state): #get the Q value for each action for passed state
        q_vals = self.q_net(state)
        return q_vals

    def greedy_action(self, state): #get the action that makes Q(state) maximum
        q_vals = self.get_qvals(state)
        greedy_a = torch.max(q_vals, dim=-1)[1].item()
        return greedy_a

##---- Prioritized Experience Replay Buffer ----##
class PrioritizedExperienceReplayBuffer():
    def __init__(self, buffer_size = 100000, burn_in = 20000, alpha = 0.1, eps = 1e-2, beta = 0.1,device=torch.device('cpu')):
        self.memory_size = buffer_size # Size capacity of the replay buffer
        self.burn_in = burn_in # Initial samples in the replay buffer
        self.alpha = alpha # Exponent to compute priorities
        self.eps = eps # Constant to add to TD error different from epsilon in epsilon-greedy
        self.beta = beta # Exponent to compute weights

        #Buffer of transitions
        self.state = torch.empty(buffer_size, *(4, 84, 84), dtype=torch.float) 
        self.action = torch.empty(buffer_size, 1, dtype=torch.int64)
        self.reward = torch.empty(buffer_size,1, dtype=torch.float)
        self.done = torch.empty(buffer_size,1, dtype=torch.int)
        self.next_state = torch.empty(buffer_size, *(4, 84, 84), dtype=torch.float)
        self.priorities = np.array([], dtype=float) # buffer of priorities
        self.max_priority = eps # Initialize max priority to epsilon (max_priority is the highest value of priorities that will be assigned to new added samples)
        self.count = 0
        self.real_size = 0
        self.device=device

    def burn_in_capacity(self):
        return self.real_size / self.burn_in 

    def change_beta(self,beta):
        self.beta=beta

    #updating priorities for sampled transitions
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if isinstance(priority, np.ndarray):
                priority = priority.item()
            self.priorities[idx] = (priority+self.eps) ** self.alpha #updating priorities with new TD-errors
            self.max_priority = max(self.max_priority, self.priorities[idx]) #keeping track of the max priority between all samples in the buffer

    #add new transition to the buffer
    def add(self, state, action, reward, done, next_state):
        if len(self.priorities) == self.memory_size: #if we have reached the maximum capacity, we start to override the buffer in a cicular fashion
            self.priorities[self.count]=float(self.max_priority)
        else:
            self.priorities=np.append(self.priorities,float(self.max_priority)) #we set as initial priority for the new stored sample with the maximum priority between all old stored samples
        self.state[self.count] = torch.as_tensor(state) 
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.memory_size #circular counter
        self.real_size = min(self.memory_size, self.real_size + 1) #real number of samples in the buffer
    
    def sample(self, batch_size):
        # Compute the probabilities of sampling each sample
        priorities = self.priorities
        probs = priorities/priorities.sum()

        #Sample a batch of indices (that are the positions of the samples in the buffer) based on their probabilities
        sample_idxs = np.random.choice(len(self.priorities), size=batch_size, p=probs) 
        
        
        batch = [
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device)
        ]

        # Compute of weights
        weights = (probs[sample_idxs] * self.real_size) ** (-self.beta)
        weights /= weights.max()  # Normalize weights 

        return batch, sample_idxs, weights
