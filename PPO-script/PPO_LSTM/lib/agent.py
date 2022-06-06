
import os
import numpy as np
from lib.ppoMemory import PPOMemory
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def boolstr_to_floatstr(v):
    if v == 'True':
        return '1'
    elif v == 'False':
        return '0'
    else:
        return v

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    T.nn.init.orthogonal_(layer.weight, std)
    T.nn.init.constant_(layer.bias, bias_const)
    return layer 


class Agent(nn.Module):
    def __init__(self, writer,n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.1, mini_batch_size=64, n_epochs=10, c1 = 0.75, c2 = 0.01,eps=1e-5,device='cpu'):
        super(Agent, self).__init__()
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(input_dims[0], 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=2, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=2, stride=1)),
            nn.ReLU(),
            nn.Flatten(1,-1),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        ) 

        self.lstm = nn.LSTM(128, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0) 

        self.actor = nn.Sequential(
            layer_init(nn.Linear(128, n_actions),std=0.01)
        )

        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 1),std=1.0)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, eps=eps)
        self.memory = PPOMemory(mini_batch_size)
        self.writer=writer
        self.c1 = c1
        self.c2 = c2
        self.device = device
        self.checkpoint_file = 'nets/PPO_LSTM.pt'
        if  os.path.exists('nets/PPO_LSTM_init.pt'):
            self.load_models(file='nets/PPO_LSTM_init.pt')
        
    def get_states(self, observation, lstm_state, done):
        hidden = self.network(observation)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = T.flatten(T.cat(new_hidden), 0, 1)
        self.last_lstm_state = lstm_state
        return new_hidden, lstm_state


    def choose_action(self, observation, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(observation, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        value = self.critic(hidden)
        return action, probs.log_prob(action), probs.entropy(), value, lstm_state

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        T.save({'model_state_dict': self.state_dict(),'optimizer_state_dict': self.optimizer.state_dict()}, self.checkpoint_file)


    def load_models(self,file = None):
        print('... loading models ...')
        if file == None:
            file = self.checkpoint_file
        checkpoint = T.load(file)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def learn(self,global_step,lstm_state):
        for _ in range(self.n_epochs):
            with T.no_grad():
                state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

                values = vals_arr
                advantage = np.zeros(len(reward_arr), dtype=np.float32)

                for t in range(len(reward_arr)-1):
                    discount = 1
                    a_t = 0
                    for k in range(t, len(reward_arr)-1):
                        a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                                (1-int(dones_arr[k])) - values[k])
                        discount *= self.gamma*self.gae_lambda

                    advantage[t] = a_t
                advantage = T.tensor(advantage).to(self.device)

            for batch in batches:
                states  = T.FloatTensor(state_arr[batch]).to(self.device)
                old_probs = T.FloatTensor(old_prob_arr[batch]).to(self.device)
                dones = T.FloatTensor(dones_arr[batch]).to(self.device)
                actions = T.FloatTensor(action_arr[batch]).to(self.device)
                _ ,new_probs, entropy, critic_value, _ =self.choose_action(states,lstm_state,dones,actions)
      
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + T.FloatTensor(values[batch]).to(self.device)
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                entropy_loss = entropy.mean()

                total_loss = actor_loss + self.c1*critic_loss - self.c2*entropy_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('losses/policy_loss',actor_loss.item(),global_step)
                self.writer.add_scalar('losses/value_loss',critic_loss.item(),global_step)
                self.writer.add_scalar('losses/entropy',entropy_loss.item(),global_step)
                self.writer.add_scalar('losses/total_loss',total_loss.item(),global_step)
        self.memory.clear_memory()