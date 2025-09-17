import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(2096442)
import cv2
from copy import deepcopy
import random
import matplotlib.pyplot as plt

class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        print(self.device)
        print("Other_Implementation")
        self.skip_frames=64
        self.stacked_frames=4
        self.count_frames=0
        self.last_frames =np.zeros((self.stacked_frames, *(84,84)), dtype=np.float32)
        self.network = Q_Net().to(self.device)
        self.initialize()
        self.step_count = 0
        self.episode = 0

    def act(self, state):
        if self.count_frames<self.skip_frames - self.stacked_frames:
            self.count_frames+=1
            return 0

        if self.count_frames<self.skip_frames:
            s=self.preprocess(state)
            j=self.count_frames%(self.skip_frames - self.stacked_frames)
            self.last_frames[j]=s
            self.count_frames+=1
            return 0

        ##A REGIME
        s=self.preprocess(state)

        self.last_frames[:-1] = self.last_frames[1:]  # Sposta gli stati di una posizione verso l'inizio
        self.last_frames[-1] = s
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
    
    def super_restart(self,env):
        self.count_frames=0
        _,_=env.reset()
        for i in range(self.skip_frames):
            if self.count_frames<self.skip_frames - self.stacked_frames:
                s, r, terminated, truncated, info = env.step(0)
                self.count_frames+=1
                continue

            if self.count_frames<self.skip_frames:
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
        truncation=False
        #simulate action
        s_1, r, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        s_1=self.preprocess(s_1)
        self.last_frames[:-1] = self.last_frames[1:]  # Sposta gli stati di una posizione verso l'inizio
        self.last_frames[-1] = deepcopy(s_1)

        self.step_count += 1
        if r >=0:
            self.actual_patience = 0
        else:
            self.actual_patience+= 1
            if self.actual_patience == self.max_patience:
                truncation = True
                self.actual_patience = 0
                r=-100
        #put experience in the buffer
        self.buffer.add(deepcopy(self.s_0), action, r, int(done), deepcopy(self.last_frames))

        self.rewards += r

        self.s_0 = deepcopy(self.last_frames)

        if done or truncation:
            self.env=self.super_restart(self.env)
            self.actual_patience = 0
            done=True
        return done

    # Implement DDQN training algorithm
    def train(self):
        self.env= gym.make('CarRacing-v2', continuous=self.continuous,max_episode_steps=930)#, render_mode='human')
        self.gamma = 0.99
        self.max_patience=100
        self.actual_patience=0
        self.epsilon = 0.7 #0.7
        
        self.learning_rate=1e-4
        self.target_network = deepcopy(self.network).to(self.device)
        self.network.set_learning_rate(self.learning_rate)
        self.network_update_frequency=10 
        self.network_sync_frequency=100
    
        self.window = 50
        self.best_mean_reward=-1000
        self.max_episodes=3000
        self.reward_threshold =600

        self.batch_size = 64
        self.exponent_alpha=0.6
        self.exponent_beta_initial=0.4
        self.exponent_beta_final=1
        self.buffer = PrioritizedExperienceReplayBuffer(state_size=(4, 84, 84), alpha=self.exponent_alpha, beta=self.exponent_beta_initial,burn_in=20000,device=self.device,buffer_size=int(50000))
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
            actual_beta=self.exponent_beta_initial + (self.exponent_beta_final - self.exponent_beta_initial) * (ep / (self.max_episodes - 1))
            print(actual_beta)
            self.buffer.change_beta(actual_beta)
            while not done:
                p = np.random.random()
                if p < self.epsilon:
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
                    if self.epsilon >= 0.05:
                        self.epsilon = self.epsilon * 0.99
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

    def update(self):
        self.network.optimizer.zero_grad()
        batch,weights,idxs = self.buffer.sample(batch_size=self.batch_size) #riga 122
        loss, td_error= self.calculate_loss(batch,weights=weights) #riga 123 (il metodo è sopra)
        self.buffer.update_priorities(idxs, td_error.cpu().numpy()) #riga 125

    def calculate_loss(self, batch, weights=None):
        #extract info from batch
        states, actions, rewards, dones, next_states = batch[0],batch[1],batch[2],batch[3],batch[4]
        #weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        weights = weights.to(self.device)

        ###############
        # DDQN Update #
        ###############
        # Q(s,a) = ??
        # qvals = self.network.get_qvals(states) #questo è Q
        # qvals = torch.gather(qvals, 1, actions)

        # # target Q(s,a) = ??
        # next_qvals= self.target_network.get_qvals(next_states)
        # next_qvals_max = torch.max(next_qvals, dim=-1)[0].reshape(-1, 1)
        # target_qvals = rewards + (1 - dones)*self.gamma*next_qvals_max #questo è Q_target

        qvals = self.network.get_qvals(states) #questo è Q
        qvals = torch.gather(qvals, 1, actions)
        next_qvals=self.network.get_qvals(next_states)
        target_next_qvals= self.target_network.get_qvals(next_states)
        max_indices=torch.max(next_qvals, dim=-1)[1]
        selected_target_values = target_next_qvals[torch.arange(target_next_qvals.size(0)), max_indices].reshape(-1, 1)
        #print(selected_target_values.shape)
        #next_qvals_max = torch.max(next_qvals, dim=-1)[0].reshape(-1, 1) ##questa è sbagliata l'azione la deve scegliere q_net secondo il suo massimo e non la target.
        target_qvals = rewards + (1 - dones)*self.gamma*selected_target_values #questo è Q_target

        assert qvals.shape == target_qvals.shape, f"{qvals.shape}, {target_qvals.shape}"
        # loss = self.loss_function( Q(s,a) , target_Q(s,a))
        #loss = self.loss_function(qvals, target_qvals)
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
        plt.savefig('mean_training_rewards_sumtree_50000.png')  
        plt.show() 
        plt.clf() 


    def forward(self, x):
        # TODO
        return x

    def save(self):
        #save_path = '/content/drive/My Drive/Reinforcement_Learning_HW/'
        #os.makedirs(save_path, exist_ok=True)
        torch.save(self.network.q_net.state_dict(),'model_buffer_sumtree_50000_final.pt')

    def intermediate_save(self):
        torch.save(self.network.q_net.state_dict(),'model_buffer_sumtree_50000_intermediate.pt')

    def load(self):
        #save_path = '/content/drive/My Drive/Reinforcement_Learning_HW/'
        #os.makedirs(save_path, exist_ok=True)
        self.network.q_net.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

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


class Q_Net(nn.Module):
    def __init__(self, lr = 1e-4, dqn = True):
        super(Q_Net, self).__init__()
        self.q_net = DQN()
        print("Q_network: ", self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def set_learning_rate(self,lr):
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def get_qvals(self, state):
        q_vals = self.q_net(state)
        return q_vals

    def greedy_action(self, state):
        q_vals = self.get_qvals(state)
        greedy_a = torch.max(q_vals, dim=-1)[1].item()
        return greedy_a

##---- Prioritized Experience Replay Buffer ----##
class PrioritizedExperienceReplayBuffer:
    def __init__(self, state_size, action_size=1, buffer_size=int(1e5), burn_in=20000, eps=1e-2, alpha=0.1, beta=0.1,device=torch.device('cpu')):
        self.eps = eps
        self.device=device
        self.tree=SumTree(buffer_size)

        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, done, next_state
        self.state = torch.empty(buffer_size, *(4, 84, 84), dtype=torch.float)
        self.action = torch.empty(buffer_size, 1, dtype=torch.int64)
        self.reward = torch.empty(buffer_size,1, dtype=torch.float)
        self.done = torch.empty(buffer_size,1, dtype=torch.int)
        self.next_state = torch.empty(buffer_size, *state_size, dtype=torch.float)

        self.count = 0
        self.real_size = 0
        self.burn_in=burn_in
        self.capacity = buffer_size

    def burn_in_capacity(self):
        return self.real_size / self.burn_in

    def change_beta(self,beta):
        self.beta=beta

    def capacity_(self):
        return self.real_size / self.capacity


    def sample(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)
            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()
        batch = [
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device)
        ]
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha
            priority=priority.item()
            self.tree.update(data_idx, priority) 
            self.max_priority = max(self.max_priority, priority)

    def add(self, state, action, reward, done, next_state):
        #state, action, reward, next_state, done = transition

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state) 
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.capacity
        self.real_size = min(self.capacity, self.real_size + 1)

class SumTree:
    def __init__(self,capacity):
        self.capacity = capacity #num of transiction. I
        self.nodes = [0] * (2 * capacity - 1)
        self.data = [None] * capacity 
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.capacity - 1  # child index in tree array
        change = value - self.nodes[idx]
        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data): 
        self.data[self.count] = data
        self.update(self.count, value)
        self.count = (self.count + 1) % self.capacity 
        self.real_size = min(self.capacity, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0 
        while 2 * idx + 1 < len(self.nodes):
            child_left, child_right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[child_left]:
                idx = child_left
            else:
                idx = child_right
                cumsum = cumsum - self.nodes[child_left]

        data_idx = idx - self.capacity + 1
        return data_idx, self.nodes[idx], self.data[data_idx]
