# FedAdapt: RL training for FedAdapt agent

We provide the configration and source code of training the RL agent in our paper. If you want to reproduce the RL training results or train the RL agent based on your own devices, please refer to the RL design in our paper.

#### Set up RL configtation

All RL configuration options are given in `../config.py` as follows:

```
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
```

#### Launch RL server

```
python RL_serverrun.py
```

#### Launch RL clients for each device

```
python RL_clientrun.py
```
