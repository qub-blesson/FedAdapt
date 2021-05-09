# FedAdapt

### FedAdapt: Adaptive Offloading of FederatedLearning on IoT Devices

This repository includes source code for the paper "FedAdapt: Adaptive Offloading of FederatedLearning on IoT Devices".

#### Code Structure

The code is composed of two parts: 
1. FL training code using FedAdapt in `FL_training` folder.
2. RL training code for FedAdapt agent in `RL_training` folder.

The results are saved as pickle files in the `results` folder. 

All configuration options are given in `config.py` , which contains the architecture, model, FL training hyperparameters and RL training hyperparameters.

Currently, the supported datasets is CIFAR10, and the supported model is CNN. The code can be extended to support other datasets and models too.

#### Setting up the environment

The code is tested on Python 3 with Pytorch version 1.4 and torchvision 0.5. In order to test the code, you need to install Pytorch and torchvision on each IoT device (Raspberry Pi and Jetson). The simplest way is to install from pre-built PyTorch and torchvision pip wheel. Please download respective pip wheel as follow:
- Pyotrch: https://github.com/FedML-AI/FedML-IoT/tree/master/pytorch-pkg-on-rpi
- Jetson: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048

Then, modify respective hostname and ip address in `config.py`. CLIENTS_CONFIG and CLIENTS_LIST in `config.py` are used for indexing and sorting.

```
# Network configration
SERVER_ADDR= '192.168.0.10'
SERVER_PORT = 51000

K = 5 # Number of devices
G = 3 # Number of groups

# Unique clients order
HOST2IP = {'pi41':'192.168.0.14' , 'pi42':'192.168.0.15', 'jetson-desktop':'192.168.0.25' , 'pi31':'192.168.0.26', 'pi32':'192.168.0.29'}
CLIENTS_CONFIG= {'192.168.0.14':0, '192.168.0.15':1, '192.168.0.25':2, '192.168.0.26':3, '192.168.0.29':4}
CLIENTS_LIST= ['192.168.0.14', '192.168.0.15', '192.168.0.25', '192.168.0.26', '192.168.0.29'] 
```

Finally, download the CIFAR10 datasets manually and put them into the `datasets/CIFAR10` folder (python version). 
- CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html

To test the code: 
- Run FL training using FedAdapt: please follow instructions in `FL_training` folder.
- Run RL training for FedAdapt agent: please follow instructions in `RL_training` folder.

#### Citation

When using this code for scientific publications, please kindly cite the above paper.

