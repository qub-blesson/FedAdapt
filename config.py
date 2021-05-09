import sys

# Network configration
SERVER_ADDR= '192.168.0.10'
SERVER_PORT = 51000

K = 5 # Number of devices
G = 3 # Number of groups

# Unique clients order
HOST2IP = {'pi41':'192.168.0.14' , 'pi42':'192.168.0.15', 'jetson-desktop':'192.168.0.25' , 'pi31':'192.168.0.36', 'pi32':'192.168.0.29'}
CLIENTS_CONFIG= {'192.168.0.14':0, '192.168.0.15':1, '192.168.0.25':2, '192.168.0.36':3, '192.168.0.29':4}
CLIENTS_LIST= ['192.168.0.14', '192.168.0.15', '192.168.0.25', '192.168.0.36', '192.168.0.29'] 

# Dataset configration
dataset_name = 'CIFAR10'
home = sys.path[0].split('FedAdapt')[0] + 'FedAdapt'
dataset_path = home +'/dataset/'+ dataset_name +'/'
N = 50000 # data length


# Model configration
model_cfg = {
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
	'VGG5' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0), 
	('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0), 
	('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
	('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
	('D', 128, 10, 1, 10, 128*10)]
}
model_name = 'VGG5'
model_size = 1.28
model_flops = 32.902
total_flops = 8488192
split_layer = [6,6,6,6,6] #Initial split layers
model_len = 7


# FL training configration
R = 100 # FL rounds
LR = 0.01 # Learning rate
B = 100 # Batch size


# RL training configration
max_episodes = 100         # max training episodes
max_timesteps = 100        # max timesteps in one episode
exploration_times = 20	   # exploration times without std decay
n_latent_var = 64          # number of variables in hidden layer
action_std = 0.5           # constant std for action distribution (Multivariate Normal)
update_timestep = 10       # update policy every n timesteps
K_epochs = 50              # update policy for K epochs
eps_clip = 0.2             # clip parameter for PPO
rl_gamma = 0.9             # discount factor
rl_b = 100				   # Batchsize
rl_lr = 0.0003             # parameters for Adam optimizer
rl_betas = (0.9, 0.999)
iteration = {'192.168.0.14' : 5, '192.168.0.15' : 5, '192.168.0.25': 50, '192.168.0.36': 5, '192.168.0.29': 5}  # infer times for each device

random = True
random_seed = 0