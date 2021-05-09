

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
import tqdm
import random
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Communicator import *
import utils
import config

np.random.seed(0)
torch.manual_seed(0)

class Sever(Communicator):
	def __init__(self, index, ip_address, server_port, model_name):
		super(Sever, self).__init__(index, ip_address)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.port = server_port
		self.model_name = model_name
		self.sock.bind((self.ip, self.port))
		self.client_socks = {}

		while len(self.client_socks) < config.K:
			self.sock.listen(5)
			logger.info("Waiting Incoming Connections.")
			(client_sock, (ip, port)) = self.sock.accept()
			logger.info('Got connection from ' + str(ip))
			logger.info(client_sock)
			self.client_socks[str(ip)] = client_sock

		self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)

		self.transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
		self.testset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=False, download=False, transform=self.transform_test)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=4)
 		
	def initialize(self, split_layers, offload, first, LR):
		if offload or first:
			self.split_layers = split_layers
			self.nets = {}
			self.optimizers= {}
			for i in range(len(split_layers)):
				client_ip = config.CLIENTS_LIST[i]
				if split_layers[i] < len(config.model_cfg[self.model_name]) -1: # Only offloading client need initialize optimizer in server
					self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg)

					#offloading weight in server also need to be initialized from the same global weight
					cweights = utils.get_model('Client', self.model_name, split_layers[i], self.device, config.model_cfg).state_dict()
					pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
					self.nets[client_ip].load_state_dict(pweights)

					self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
					  momentum=0.9)
				else:
					self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg)
			self.criterion = nn.CrossEntropyLoss()

		msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)

	def train(self, thread_number, client_ips):
		# Network test
		self.net_threads = {}
		for i in range(len(client_ips)):
			self.net_threads[client_ips[i]] = threading.Thread(target=self._thread_network_testing, args=(client_ips[i],))
			self.net_threads[client_ips[i]].start()

		for i in range(len(client_ips)):
			self.net_threads[client_ips[i]].join()

		self.bandwidth = {}
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_TEST_NETWORK')
			self.bandwidth[msg[1]] = msg[2]

		# Training start
		self.threads = {}
		for i in range(len(client_ips)):
			if config.split_layer[i] == (config.model_len -1):
				self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_no_offloading, args=(client_ips[i],))
				logger.info(str(client_ips[i]) + ' no offloading training start')
				self.threads[client_ips[i]].start()
			else:
				logger.info(str(client_ips[i]))
				self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_offloading, args=(client_ips[i],))
				logger.info(str(client_ips[i]) + ' offloading training start')
				self.threads[client_ips[i]].start()

		for i in range(len(client_ips)):
			self.threads[client_ips[i]].join()

		self.ttpi = {} # Training time per iteration
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_TRAINING_TIME_PER_ITERATION')
			self.ttpi[msg[1]] = msg[2]

		self.group_labels = self.clustering(self.ttpi, self.bandwidth)
		self.offloading = self.get_offloading(self.split_layers)
		state = self.concat_norm(self.ttpi, self.offloading)

		return state, self.bandwidth

	def _thread_network_testing(self, client_ip):
		msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK')
		msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
		self.send_msg(self.client_socks[client_ip], msg)

	def _thread_training_no_offloading(self, client_ip):
		pass

	def _thread_training_offloading(self, client_ip):
		iteration = int((config.N / (config.K * config.B)))
		for i in range(iteration):
			msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
			smashed_layers = msg[1]
			labels = msg[2]

			inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
			self.optimizers[client_ip].zero_grad()
			outputs = self.nets[client_ip](inputs)
			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizers[client_ip].step()

			# Send gradients to client
			msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'+str(client_ip), inputs.grad]
			self.send_msg(self.client_socks[client_ip], msg)

		logger.info(str(client_ip) + ' offloading training end')
		return 'Finish'

	def aggregate(self, client_ips):
		w_local_list =[]
		for i in range(len(client_ips)):
			msg = self.recv_msg(self.client_socks[client_ips[i]], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
			if config.split_layer[i] != (config.model_len -1):
				w_local = (utils.concat_weights(self.uninet.state_dict(),msg[1],self.nets[client_ips[i]].state_dict()),config.N / config.K)
				w_local_list.append(w_local)
			else:
				w_local = (msg[1],config.N / config.K)
				w_local_list.append(w_local)
		zero_model = utils.zero_init(self.uninet).state_dict()
		aggregrated_model = utils.fed_avg(zero_model, w_local_list, config.N)
		
		self.uninet.load_state_dict(aggregrated_model)
		return aggregrated_model

	def test(self, r):
		self.uninet.eval()
		test_loss = 0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.testloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.uninet(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

		acc = 100.*correct/total
		logger.info('Test Accuracy: {}'.format(acc))

		# Save checkpoint.
		torch.save(self.uninet.state_dict(), './'+ config.model_name +'.pth')

		return acc

	def clustering(self, state, bandwidth):
		#sort bandwidth in config.CLIENTS_LIST order
		bandwidth_order =[]
		for c in config.CLIENTS_LIST:
			bandwidth_order.append(bandwidth[c])

		labels = [0,0,1,0,0] # Previous clustering results in RL
		for i in range(len(bandwidth_order)):
			if bandwidth_order[i] < 5:
				labels[i] = 2 # If network speed is limited under 5Mbps, we assign the device into group 2

		return labels

	def adaptive_offload(self, agent, state):
		action = agent.exploit(state)
		action = self.expand_actions(action, config.CLIENTS_LIST)

		config.split_layer = self.action_to_layer(action)
		logger.info('Next Round OPs: ' + str(config.split_layer))

		msg = ['SPLIT_LAYERS',config.split_layer]
		self.scatter(msg)
		return config.split_layer

	def expand_actions(self, actions, clients_list): # Expanding group actions to each device
		full_actions = []

		for i in range(len(clients_list)):
			full_actions.append(actions[self.group_labels[i]])

		return full_actions

	def action_to_layer(self, action): # Expanding group actions to each device
		#first caculate cumulated flops
		model_state_flops = []
		cumulated_flops = 0

		for l in config.model_cfg[config.model_name]:
			cumulated_flops += l[5]
			model_state_flops.append(cumulated_flops)

		model_flops_list = np.array(model_state_flops)
		model_flops_list = model_flops_list / cumulated_flops

		split_layer = []
		for v in action:
			idx = np.where(np.abs(model_flops_list - v) == np.abs(model_flops_list - v).min()) 
			idx = idx[0][-1]
			if idx >= 5: # all FC layers combine to one option
				idx = 6
			split_layer.append(idx)
		return split_layer

	def concat_norm(self, ttpi, offloading):
		ttpi_order = []
		offloading_order =[]
		for c in config.CLIENTS_LIST:
			ttpi_order.append(ttpi[c])
			offloading_order.append(offloading[c])

		group_max_index = [0 for i in range(config.G)]
		group_max_value = [0 for i in range(config.G)]
		for i in range(len(config.CLIENTS_LIST)):
			label = self.group_labels[i]
			if ttpi_order[i] >= group_max_value[label]:
				group_max_value[label] = ttpi_order[i]
				group_max_index[label] = i

		ttpi_order = np.array(ttpi_order)[np.array(group_max_index)]
		offloading_order = np.array(offloading_order)[np.array(group_max_index)]
		state = np.append(ttpi_order, offloading_order)
		return state

	def get_offloading(self, split_layer):
		offloading = {}
		workload = 0

		assert len(split_layer) == len(config.CLIENTS_LIST)
		for i in range(len(config.CLIENTS_LIST)):
			for l in range(len(config.model_cfg[config.model_name])):
				if l <= split_layer[i]:
					workload += config.model_cfg[config.model_name][l][5]
			offloading[config.CLIENTS_LIST[i]] = workload / config.total_flops
			workload = 0

		return offloading


	def reinitialize(self, split_layers, offload, first, LR):
		self.initialize(split_layers, offload, first, LR)

	def scatter(self, msg):
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)
