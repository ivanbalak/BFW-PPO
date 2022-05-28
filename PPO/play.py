#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import torch as T
from torch.distributions.categorical import Categorical
from lib import actorNetwork
from lib import utils, wrappers

from tensorboardX import SummaryWriter

if __name__ == '__main__':
    utils.kill_game_processes()
    env = wrappers.make_env("Bfw-v0", gui=True, scenario="side2_pass_3units", variations=3, rotation=0)
    alpha = 0.0001
    
    writer = SummaryWriter()
    n_actions=env.action_space.n
    input_dims=env.observation_space.shape
    actor=actorNetwork.ActorNetwork(n_actions,  input_dims, alpha)
    actor.load_checkpoint()
    actor.eval()
    observation = env.reset()
    done = False
    score = 0
    move = 0
    while not done:
        state = T.tensor([observation], dtype=T.float).to(actor.device)
        dist = actor(state)
        action = dist.sample()
        action = T.squeeze(action).item()
        observation, reward, done, info = env.step((0, action))
        score += reward
        move += 1
        print('move',move,', action',action,',score %.1f'% score)