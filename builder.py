from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *

def get_optimizer_from_arguments(config: dict, params: Iterable[nn.Parameter]) -> optim.Optimizer:
    """
    Get the optimizer from the given argument
    :param optim_config: the optimization dictionary
    :param params: the model parameters to be optimized by the optimizers
    :return: the correctly set optimizer.
    """
    optim_config = config['training']['optimizer']
    optim_name = optim_config["name"].lower()
    if optim_name == "sgd":
        optimizer = optim.SGD(params,
                              lr=optim_config["learning_rate"],
                              momentum=optim_config["momentum"],
                              weight_decay=optim_config["weight_decay"],
                              nesterov=optim_config["nesterov"])
    elif optim_name == "adam":
        optimizer = optim.Adam(params,
                               lr=optim_config["learning_rate"],
                               betas=tuple(optim_config["betas"]),
                               weight_decay=optim_config["weight_decay"])
    elif optim_name == "rmsprop":
        optimizer = optim.RMSprop(params,
                                  lr=optim_config["learning_rate"],
                                  weight_decay=optim_config["weight_decay"])
    elif optim_name == "adamw":
        optimizer = optim.AdamW(params,
                               lr=optim_config["learning_rate"],
                               betas=tuple(optim_config["betas"]),
                               weight_decay=optim_config["weight_decay"],
					amsgrad=True)
    else:
        raise KeyError("The given optimizer is not supported {}".format(optim_config["name"]))

    return optimizer


def get_lr_scheduler_from_arguments(config: dict, optimizer: optim.Optimizer): \
    # -> Union[StepLR, CyclicLR, MultiStepLR, ExponentialLR, LambdaLR, dict]:
    """
    Get the learning rate scheduler from the given argument
    :param scheduler_config: the config dictionary for the learning rate scheduler.
    :param optimizer: the optimizer to be wrapped up in the learning rate scheduler
    :return: the learning rate scheduler
    """
    scheduler_config = config['training']['lr_scheduler']

    # get the scheduler name
    scheduler_name = scheduler_config["name"].lower()

    if scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=scheduler_config["nb_step"], gamma=scheduler_config["decay_factor"])
    elif scheduler_name == "clr":
        scheduler = CyclicLR(optimizer, base_lr=scheduler_config["base_lr"], max_lr=scheduler_config["max_lr"])
    elif scheduler_name == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=scheduler_config["milestones"],
                                gamma=scheduler_config["decay_factor"])
    elif scheduler_name == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=scheduler_config["decay_factor"])
    elif scheduler_name == "constant":
        scheduler = LambdaLR(optimizer, lambda epoch: 1.0)
    elif scheduler_name == "plateau":
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode=scheduler_config["mode"],
                                           patience=scheduler_config["patience"],
                                           factor=scheduler_config["decay_factor"]),
            # might need to change here
            'monitor': scheduler_config["monitor"],  # Default: val_loss
            'reduce_on_plateau': False,  # For ReduceLROnPlateau scheduler, default
            # 'interval': 'step',
            'interval': 'epoch',
            # need to change here
            # 'frequency': 300
            'frequency': 1
        }
    else:
        raise KeyError(f"The given learning rate scheduler is not recognised ({scheduler_name})")
    return scheduler


def get_accelerator_device_from_args(config: dict): \
    # -> Tuple[str, Union[str, int, List[int]]]:
    """
    Get the hardware acceleration and device from the arguments
    :param config: the configuration dictionary with all the options
    :return: a tuple accelerator, device set from the arguments.
    """
    hardware_config = config['hardware']

    # Choose the specific GPUs or the number of them to where we have to train the model
    if torch.cuda.is_available() and hardware_config['use_cuda']:
        # Use cuda
        device = "cuda:0"
    else:
        # Use cpu instead of GPU
        device = "cpu"

    return device
