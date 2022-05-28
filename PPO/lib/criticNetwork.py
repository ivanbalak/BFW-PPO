import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=10, fc2_dims=10,
            chkpt_dir='./tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.conv = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_dims)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )    

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        self.to(self.device)

    def _get_conv_out(self, shape):
        o = self.conv(T.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


    def save_checkpoint(self):
        T.save({'model_state_dict': self.state_dict(),'optimizer_state_dict': self.optimizer.state_dict()}, self.checkpoint_file)

    def load_checkpoint(self):
        checkpoint = T.load(self.checkpoint_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])