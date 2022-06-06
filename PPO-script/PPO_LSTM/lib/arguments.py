import argparse
import os
from distutils.util import strtobool

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    #Engine arguments
    parser.add_argument("--exp-name", type=str, default="PPO_LSTM",
        help="The name of this experiment")
    parser.add_argument("--gym-id", type=str, default="Bfw-v0",
        help="The id of the gym environment")
    parser.add_argument("--seed", type=int, default=1,
        help="Seed of the experiment random number generator")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="Toggles cuda GPU acceleration")
    parser.add_argument("--showgui", type=lambda x: bool(strtobool(x)), default=False,nargs="?", const=False,
            help="Toggles the environment GUI"),
    parser.add_argument("--variations", type=int, default=2000000,
        help="Sets the game map variations, how often the map will be updated or mutated"),
    parser.add_argument("--rotation", type=int, default=0,
        help="Sets the game map rotation, how many different maps will be included in map rotation, range 0-5"),
    parser.add_argument("--mutations", type=int, default=0,
        help="How many times to mutate the game map"),
    parser.add_argument("--map", type=str, default="maps/May28_19-37-05.map",
        help="Use map file"),        
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles `torch.backends.cudnn.deterministic`"),


    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="Total timesteps of the experiments")
    parser.add_argument("--batch-size", type=int, default=100,
        help="The size of the batch, the number of steps to run in each environment per policy rollout")   
    parser.add_argument("--mini-batch-size", type=int, default=100,
        help="The size of the mini batch, the number of steps per training iteration")   
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="The learning rate of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="The discount factor gamma for the general advantage estimation")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="The lambda for the general advantage estimation")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="The number of epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="The surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.005,
        help="Coefficient of the entropy, c2")
    parser.add_argument("--vf-coef", type=float, default=0.6,
        help="Coefficient of the value function, c1")
    parser.add_argument("--epsilon", type=float, default=1e-5,
        help="Optimizer epsilon")
  
    args = parser.parse_args()
    # fmt: on
    return args