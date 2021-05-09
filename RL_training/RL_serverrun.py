import pickle
import torch
import tqdm
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from RLEnv import Env
import config
import utils
import RLEnv
import PPO

if config.random:
	torch.manual_seed(config.random_seed)
	np.random.seed(config.random_seed)
	logger.info('Random seed: {}'.format(config.random_seed))

# Creating environment
env = Env(0, config.SERVER_ADDR, config.SERVER_PORT, config.CLIENTS_LIST, config.model_name, config.model_cfg, config.rl_b)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Creating PPO agent
state_dim = env.state_dim
action_dim = env.action_dim
memory = PPO.Memory()
ppo = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma, config.K_epochs, config.eps_clip)

# RL training
logger.info('==> RL Training Start.')
time_step = 0
update_epoch = 1

res = {}
res['rewards'], res['maxtime'], res['actions'], res['std'] = [], [], [], []

for i_episode in tqdm.tqdm(range(1, config.max_episodes+1)):
	done = False # Flag controling finish of one episode
	if i_episode == 1: # We run two times of initial state to get stable training time
		first = True
		state = env.reset(done, first)
	else:
		first = False
		state = env.reset(done, first)


	for t in range(config.max_timesteps):
		logger.info('====================================>')
		time_step +=1
		action, action_mean, std = ppo.select_action(state, memory)
		state, reward, maxtime, done = env.step(action, done)
		logger.info('Current reward: ' + str(reward))
		logger.info('Current maxtime: ' + str(maxtime))

		# Saving reward and is_terminals:
		memory.rewards.append(reward)
		memory.is_terminals.append(done)

		
		# Update
		if time_step % config.update_timestep == 0:
			ppo.update(memory)
			logger.info('Agent has been updated: ' + str(update_epoch))
			if update_epoch > config.exploration_times:
				ppo.explore_decay(update_epoch - config.exploration_times)

			memory.clear_memory()
			time_step = 0
			update_epoch += 1

			# Record the results for each update epoch
			with open('RL_res.pkl','wb') as f:
				pickle.dump(res,f)

			# save the agent every updates
			torch.save(ppo.policy.state_dict(), './PPO.pth')
		

		res['rewards'].append(reward)
		res['maxtime'].append(maxtime)
		res['actions'].append((action, action_mean))
		res['std'].append(std)

		if done:
			break

		# stop when get control update epoch
		if update_epoch > 50:
			break
