import json

import torch

from typing import Union
from dataclasses import dataclass
from collections import namedtuple

@dataclass
class Hyperparameters:
    MINIBATCH_SIZE: int = 32
    REPLAY_MEMORY_SIZE: int = 1000000
    AGENT_HISTORY_LENGTH: int = 4
    TARGET_UPDATE_FREQ: int = 10000 # number policy updates between target updates
    DISCOUNT_FACTOR: float = 0.99 # gamma
    ACTION_REPEAT: int = 4 # repeat each action during this many frames
    POLICY_UPDATE_FREQ: int = 4 # number of actions selected between policy updates
    LEARNING_RATE: float = 0.00025 # used by RMSProp
    GRADIENT_MOMENTUM: float = 0.95 # used by RMSProp
    SQUARED_GRADIENT_MOMENTUM: float = 0.95 # used by RMSProp
    MIN_SQUARED_GRADIENT: float = 0.01 # used by RMSProp
    INITIAL_EPSILON: float = 1.0 # initial value for epsilon-greedy exploration
    FINAL_EPSILON: float = 0.1 # final value for epsilon-greedy exploration
    EPSILON_DECAY: int = 1000000 # frames taken to decay epsilon value
    REPLAY_START_SIZE: int = 50000 # random policy is run for this number of frames before learning starts
    NOOP_MAX: int = 30 # maximum noops performed at the start of an episode

    @property
    def param_names(self):
        return ['MINIBATCH_SIZE','REPLAY_MEMORY_SIZE','AGENT_HISTORY_LENGTH',
                'TARGET_UPDATE_FREQ','DISCOUNT_FACTOR','ACTION_REPEAT',
                'POLICY_UPDATE_FREQ','LEARNING_RATE','GRADIENT_MOMENTUM',
                'SQUARED_GRADIENT_MOMENTUM','MIN_SQUARED_GRADIENT','INITIAL_EPSILON',
                'FINAL_EPSILON','EPSILON_DECAY','REPLAY_START_SIZE','NOOP_MAX']

    def save_hyperparameters(self, path):
        obj = {param: getattr(self, param) for param in self.param_names} 
        with open(path) as f:
            json.dump(obj, f)
        return path

    def load_hyperparameters(self, path):
        with open(path) as f:
            obj = json.load(f)
        for param in obj:
            setattr(self, param, obj[param])
        return path



@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor

    def to(self, device: Union[str, torch.device]):
        self.state = self.state.to(device=device)
        self.action = self.action.to(device=device)
        self.reward = self.reward.to(device=device)
        self.done = self.done.to(device=device)


OptimizerParams = namedtuple('OptimizerParams', ['LEARNING_RATE', 'GRADIENT_MOMENTUM', 'SQUARED_GRADIENT_MOMENTUM', 'MIN_SQUARED_GRADIENT'])