import torch
import socket
import time
import multiprocessing
import os
import argparse

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Client import Client
import config
import utils

parser=argparse.ArgumentParser()
parser.add_argument('--offload', help='FedAdapt or classic FL mode', type= utils.str2bool, default= False)
args=parser.parse_args()

ip_address = config.HOST2IP[socket.gethostname()]
index = config.CLIENTS_CONFIG[ip_address]
datalen = config.N / config.K
split_layer = config.split_layer[index]
LR = config.LR

logger.info('Preparing Client')
client = Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, datalen, 'VGG5', split_layer)

offload = args.offload
first = True # First initializaiton control
client.initialize(split_layer, offload, first, LR)
first = False 

logger.info('Preparing Data.')
cpu_count = multiprocessing.cpu_count()
trainloader, classes= utils.get_local_dataloader(index, cpu_count)

if offload:
	logger.info('FedAdapt Training')
else:
	logger.info('Classic FL Training')

flag = False # Bandwidth control flag.

for r in range(config.R):
	logger.info('====================================>')
	logger.info('ROUND: {} START'.format(r))

	'''
	# Network bandwidth changing
	if socket.gethostname() == 'jetson-desktop':
		if r == 50 and flag == False : #start from next round
			#cmd = "sudo tc qdisc add dev wlan0 root tbf rate 5mbit latency 10ms burst 1600"
			pass #Jetson needs rebuild linux kernel
		if r == 60 and flag == True : #start from next round
			#cmd = "sudo tc qdisc del dev wlan0 root"
			pass

	if socket.gethostname() == 'pi41':
		if r == 60 and flag == False : #start from next round
			cmd = "sudo tc qdisc add dev wlan0 root tbf rate 5mbit latency 10ms burst 1600"
			print(cmd)					
			os.system(cmd)
			flag = True
		if r == 70 and flag == True : #start from next round
			cmd = "sudo tc qdisc del dev wlan0 root"
			print(cmd)					
			os.system(cmd)
			flag = False

	if socket.gethostname() == 'pi42':
		if r == 70 and flag == False : #start from next round
			cmd = "sudo tc qdisc add dev wlan0 root tbf rate 5mbit latency 10ms burst 1600"
			print(cmd)					
			os.system(cmd)
			flag = True
		if r == 80 and flag == True : #start from next round
			cmd = "sudo tc qdisc del dev wlan0 root"
			print(cmd)					
			os.system(cmd)
			flag = False

	if socket.gethostname() == 'pi31':
		if r == 80 and flag == False : #start from next round
			cmd = "sudo tc qdisc add dev wlan0 root tbf rate 5mbit latency 10ms burst 1600"
			print(cmd)					
			os.system(cmd)
			flag = True
		if r == 90 and flag == True : #start from next round
			cmd = "sudo tc qdisc del dev wlan0 root"
			print(cmd)					
			os.system(cmd)
			flag = False

	if socket.gethostname() == 'pi32':
		if r == 90 and flag == False : #start from next round
			cmd = "sudo tc qdisc add dev wlan0 root tbf rate 5mbit latency 10ms burst 1600"
			print(cmd)					
			os.system(cmd)
			flag = True
		if r == 100 and flag == True : #start from next round
			cmd = "sudo tc qdisc del dev wlan0 root"
			print(cmd)					
			os.system(cmd)
			flag = False
	'''
	training_time = client.train(trainloader)
	logger.info('ROUND: {} END'.format(r))
	
	logger.info('==> Waiting for aggregration')
	client.upload()

	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
	s_time_rebuild = time.time()
	if offload:
		config.split_layer = client.recv_msg(client.sock)[1]

	if r > 49:
		LR = config.LR * 0.1

	client.reinitialize(config.split_layer[index], offload, first, LR)
	e_time_rebuild = time.time()
	logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
	logger.info('==> Reinitialization Finish')
