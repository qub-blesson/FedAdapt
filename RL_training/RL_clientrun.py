import torch
import socket
import multiprocessing
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from RLEnv import RL_Client
import config
import utils

if config.random:
	torch.manual_seed(config.random_seed)
	np.random.seed(config.random_seed)
	logger.info('Random seed: {}'.format(config.random_seed))

first = True
ip_address = config.HOST2IP[socket.gethostname()]
index = config.CLIENTS_CONFIG[ip_address]
datalen = config.N / config.K
split_layer = config.split_layer[index]

logger.info('==> Preparing Data..')
cpu_count = multiprocessing.cpu_count()
trainloader, classes= utils.get_local_dataloader(index, cpu_count)

logger.info('==> Preparing RL_Client..')
rl_client = RL_Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, datalen, config.model_name, split_layer, config.model_cfg)

while True:
	reset_flag = rl_client.recv_msg(rl_client.sock, 'RESET_FLAG')[1]
	if reset_flag:
		rl_client.initialize(len(config.model_cfg[config.model_name])-1)
	else:
		logger.info('==> Next Timestep..')
		config.split_layer = rl_client.recv_msg(rl_client.sock, 'SPLIT_LAYERS')[1]
		rl_client.reinitialize(config.split_layer[index])

	logger.info('==> Training Start..')
	if first:
		rl_client.infer(trainloader)
		rl_client.infer(trainloader)
		first = False
	else:
		rl_client.infer(trainloader)

	
	


