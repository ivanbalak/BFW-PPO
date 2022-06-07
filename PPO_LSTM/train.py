"""

Trainng PPO LSTM

 
"""


import os, time
import random
from lib.agent import Agent
from lib import wrappers, arguments
import numpy as np
import torch as T
from tensorboardX import SummaryWriter

      


if __name__ == '__main__':
    # delete all loose instances od the game
    os.system("killall -9 wesnoth")

    args = arguments.parse_args()
    if args.seed != None:
        # seeding the random number generator
        random.seed(args.seed)
        np.random.seed(args.seed)
        T.manual_seed(args.seed)
    T.backends.cudnn.deterministic = args.torch_deterministic

    run_name = f"{args.gym_id}_{args.exp_name}_{int(time.time())}"
    
    #init variables
    n_games = args.total_timesteps//100
    batch_size = args.batch_size
    mini_batch_size = args.mini_batch_size
    n_epochs = args.update_epochs
    alpha = args.learning_rate
    policy_clip=args.clip_coef
    c1 = args.vf_coef
    c2 = args.ent_coef
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    #load game gym
    env = wrappers.make_env(args.gym_id, gui=args.showgui, scenario="side2_pass_3units", variations=args.variations, mutations = args.mutations, rotation=args.rotation, map = 'maps/Jun04_06-36-00.map')
    
    # init indicators
    best_score = env.reward_range[0]
    score_history = []
    best_score = 0
    learn_iters = 0
    avg_score = 0
    global_step = 0
    done = False
    episode = 0
    local_step = 0
    
    #create a Tensorboard writer and save hyperparameters 
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    device = T.device("cuda" if T.cuda.is_available() and args.cuda else "cpu")

    
    writer.add_hparams({'learn_rate': alpha, 'batch_size': batch_size,'epochs': n_epochs, 'mini_batch':mini_batch_size, 'clip':policy_clip,'c1':c1, 'c2':c2},{'hparam/best_score' : best_score})
    
    #create AI agent
    agent = Agent(writer,n_actions=env.action_space.n, mini_batch_size=mini_batch_size,  
                    alpha=alpha, n_epochs=n_epochs,  gamma=args.gamma,gae_lambda=args.gae_lambda,
                    input_dims=env.observation_space.shape, policy_clip=policy_clip, c1 = c1, c2 = c2,eps=args.epsilon, device=device).to(device)




    #prep storage for LSTM hidden and cell states 
    lstm_num_layers =1
    lstm_hidden_size = 128
    next_done = T.FloatTensor([done]).to(device)
    next_lstm_state = (
        T.zeros(lstm_num_layers, 1, lstm_hidden_size).to(device),
        T.zeros(lstm_num_layers, 1, lstm_hidden_size).to(device),
    ) 
    

    observation = env.reset()
    #run training

    for update in range(1, num_updates + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        observation = env.reset()
        score = 0
        #roll out a batch of enviroments
        for step in range(0, args.batch_size):
            with T.no_grad():
                state = T.FloatTensor([observation]).to(device)
                action, prob, _, val,next_lstm_state = agent.choose_action(state, next_lstm_state,next_done)
                #action = action.cpu().numpy()
                observation_, reward, done, info = env.step((0, T.squeeze(action).item()))
                score += reward
                global_step += 1
                agent.remember(observation, T.squeeze(action).item(), T.squeeze(prob).item(), T.squeeze(val).item(), reward, done)
                next_done = T.FloatTensor([done]).to(device)
                if done == True:
                    episode += 1
                    observation = env.reset()
                    score_history.append(score)
                    avg_score = np.mean(score_history[-100:])
                    # keep track of the reward and store model state if it's good   
                    if avg_score > best_score:
                        best_score = avg_score
                        agent.save_models()
                    # log stats to Tensorboard
                    player_stats=env.get_stats(0)
                    writer.add_scalar("charts/invalid_moves_per_unit", player_stats.invalid_moves, global_step)
                    writer.add_scalar("charts/villages_taken", player_stats.villages_taken, global_step)
                    writer.add_scalar("charts/villages_lost", player_stats.villages_lost, global_step)
                    writer.add_scalar("charts/mean_movement_range", player_stats.mean_movement_range, global_step)
                    writer.add_scalar("charts/gold", player_stats.gold, global_step)
                    print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                        'time_steps', global_step, 'learning_steps', learn_iters)
                    writer.add_scalar("charts/episodic_return", score,global_step)
                    writer.add_scalar("charts/episodic_length", local_step, global_step)
                    writer.add_scalar('charts/learning_steps', learn_iters,global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    local_step = 0
                else:
                    observation = observation_
                    local_step += 1
                #train AI policy 
        agent.learn(global_step,initial_lstm_state)
        learn_iters += 1
            
   



       
    writer.close()