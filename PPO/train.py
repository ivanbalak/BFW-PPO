"""

Training PPO

"""


import collections
from lib import agent
from lib import utils, wrappers
import gym.spaces
import numpy as np
import torch as T
from tensorboardX import SummaryWriter

    
    


if __name__ == '__main__':

    utils.kill_game_processes()
    env = wrappers.make_env("Bfw-v0", gui=False, scenario="side2_pass_3units", variations=3, rotation=0)
    N = 64
    batch_size = 8
    n_epochs = 8
    alpha = 0.00003
    policy_clip=0.1
    c1 = 0.75
    c2 = 0.01
    writer = SummaryWriter()

    agent = agent.Agent(writer,n_actions=env.action_space.n, batch_size=batch_size,  
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape, policy_clip=policy_clip, c1 = c1, c2 = c2)
    #agent.actor.load_checkpoint()
    #agent.critic.load_checkpoint()
    #agent.actor.train()
    #agent.critic.train()
    n_games = 10000
    dummy = T.zeros(1,3,10,10)
    #writer.add_graph(agent.actor,(dummy,))
    writer.add_graph(agent.critic,(dummy,))
    best_score = env.reward_range[0]
    score_history = []
    best_score = 0
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    mirror = 0

    frame_idx1 = 0
    writer.add_hparams({'learn_rate': alpha, 'batch_size': batch_size,'epochs': n_epochs, 'N':N, 'clip':policy_clip,'c1':c1, 'c2':c2},{'hparam/best_score' : best_score})
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step((0, action))
            
            score += reward
            n_steps += 1
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_

        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        player_stats=env.get_stats(0)
        writer.add_scalar("invalid_moves_per_unit", player_stats.invalid_moves, i)
        writer.add_scalar("villages_taken", player_stats.villages_taken, i)
        writer.add_scalar("villages_lost", player_stats.villages_lost, i)
        writer.add_scalar("mean_movement_range", player_stats.mean_movement_range, i)
        writer.add_scalar("gold", player_stats.gold, i)
        state=T.tensor([observation], dtype=T.float).to(agent.actor.device)

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
        writer.add_scalar("score", score,i)
        writer.add_scalar("avg_score", avg_score,i)
     
        writer.add_scalar('time_steps', n_steps,i)
        writer.add_scalar('learning_steps', learn_iters,i)

    writer.close()