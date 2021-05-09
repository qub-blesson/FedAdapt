'''Some helper functions for FedAdapt, including:
	- get_local_dataloader: split dataset and get respective dataloader.
	- get_model: build the model according to location and split layer.
	- send_msg: send msg with type checking.
	- recv_msg: receive msg with type checking.
	- split_weights_client: split client's weights from holistic weights.
	- split_weights_server: split server's weights from holistic weights
	- concat_weights: concatenate server's weights and client's weights.
	- zero_init: zero initialization.
	- fed_avg: FedAvg aggregation.
	- norm_list: normlize each item in a list with sum.
	- str2bool.
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


import pickle, struct, socket
from vgg import *
from config import *
import collections
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

def get_local_dataloader(CLIENT_IDEX, cpu_count):
	indices = list(range(N))
	part_tr = indices[int((N/K) * CLIENT_IDEX) : int((N/K) * (CLIENT_IDEX+1))]

	transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
	trainset = torchvision.datasets.CIFAR10(
		root=dataset_path, train=True, download=False, transform=transform_train)
	subset = Subset(trainset, part_tr)
	trainloader = DataLoader(
		subset, batch_size=B, shuffle=True, num_workers=cpu_count)

	classes = ('plane', 'car', 'bird', 'cat', 'deer',
		   'dog', 'frog', 'horse', 'ship', 'truck')
	return trainloader,classes

def get_model(location, model_name, layer, device, cfg):
	cfg = cfg.copy()
	net = VGG(location, model_name, layer, cfg)
	net = net.to(device)
	logger.debug(str(net))
	return net

def send_msg(sock, msg):
	msg_pickle = pickle.dumps(msg)
	sock.sendall(struct.pack(">I", len(msg_pickle)))
	sock.sendall(msg_pickle)
	logger.debug(msg[0]+'sent to'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

def recv_msg(sock, expect_msg_type=None):
	msg_len = struct.unpack(">I", sock.recv(4))[0]
	msg = sock.recv(msg_len, socket.MSG_WAITALL)
	msg = pickle.loads(msg)
	logger.debug(msg[0]+'received from'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

	if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
		raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
	return msg

def split_weights_client(weights,cweights):
	for key in cweights:
		assert cweights[key].size() == weights[key].size()
		cweights[key] = weights[key]
	return cweights

def split_weights_server(weights,cweights,sweights):
	ckeys = list(cweights)
	skeys = list(sweights)
	keys = list(weights)

	for i in range(len(skeys)):
		assert sweights[skeys[i]].size() == weights[keys[i + len(ckeys)]].size()
		sweights[skeys[i]] = weights[keys[i + len(ckeys)]]

	return sweights

def concat_weights(weights,cweights,sweights):
	concat_dict = collections.OrderedDict()

	ckeys = list(cweights)
	skeys = list(sweights)
	keys = list(weights)

	for i in range(len(ckeys)):
		concat_dict[keys[i]] = cweights[ckeys[i]]

	for i in range(len(skeys)):
		concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]

	return concat_dict



def zero_init(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			init.zeros_(m.weight)
			if m.bias is not None:
				init.zeros_(m.bias)
		elif isinstance(m, nn.BatchNorm2d):
			init.zeros_(m.weight)
			init.zeros_(m.bias)
			init.zeros_(m.running_mean)
			init.zeros_(m.running_var)
		elif isinstance(m, nn.Linear):
			init.zeros_(m.weight)
			if m.bias is not None:
				init.zeros_(m.bias)
	return net

def fed_avg(zero_model, w_local_list, totoal_data_size):
	keys = w_local_list[0][0].keys()
	
	for k in keys:
		for w in w_local_list:
			beta = float(w[1]) / float(totoal_data_size)
			if 'num_batches_tracked' in k:
				zero_model[k] = w[0][k]
			else:	
				zero_model[k] += (w[0][k] * beta)

	return zero_model

def norm_list(alist):	
	return [l / sum(alist) for l in alist]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
