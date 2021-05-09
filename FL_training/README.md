# FedAdapt: FL training using FedAdapt

#### Run FL training using FedAdapt

We provide the trained PPO agent (`PPO_FedAdapt.pth`) for the part 3 of experiments in our paper. You could run FL training using FedAdapt by setting ``` --offload True ``` or classic FL by setting ``` -- offload False ```.

#### Launch FedAdapt server

```
python FedAdapt_serverrun.py --offload True #FedAdapt training
python FedAdapt_serverrun.py --offload False #Classic FL training
```

#### Launch FedAdapt client for each device

```
python FedAdapt_clientrun.py --offload True #FedAdapt training
python FedAdapt_clientrun.py --offload False #Classic FL training
```