import os
import json
import random

import torch
from torch import optim

from typing import Tuple

import logging
from datetime import datetime

from utils import Hyperparameters, OptimizerParams, Transition
from dqn_model import DQN_Model

class LearningAgent():
    def __init__(self, num_actions: int, state_shape: Tuple[int, int, int] = (4,84,84), \
                 gamma = 0.99, optimizer_params: OptimizerParams = (0.00025, 0.95, 0.95, 0.01), \
                 name = 'default_name', root = './data', device = 'cpu'):
        self.logger = logging.getLogger(f"{name} ({self.__class__.__name__})")

        self.name = name
        self.folder = os.path.join(root, name)
        self.device = device

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.gamma = gamma
        self.optimizer_params = optimizer_params

        # Create DQNs and copy state dict of policy_net to target_net
        self.policy_net = DQN_Model(state_shape[0], state_shape[1:], num_actions).to(device)
        self.target_net = DQN_Model(state_shape[0], state_shape[1:], num_actions).to(device)
        self.update_target()
        self.eval()

        if not isinstance(optimizer_params, OptimizerParams):
            optimizer_params = OptimizerParams(*optimizer_params)

        # Create optimizer
        self.optimizer = optim.RMSprop(
            params=self.policy_net.parameters(),
            lr=optimizer_params.LEARNING_RATE,
            momentum=optimizer_params.GRADIENT_MOMENTUM,
            alpha=optimizer_params.SQUARED_GRADIENT_MOMENTUM,
            eps=optimizer_params.MIN_SQUARED_GRADIENT
        )

        self.logger.info(f"Initialized learning agent {self.name}")

    def load_agent(self, folder: str = None):
        if folder is None:
            folder = self.folder
        self.logger.info(f"Loading agent from {folder}")

        file_exists = lambda *args : os.path.isfile(os.path.join(*args))
        files_exist = lambda files : all([file_exists(folder, f) for f in files])

        if not os.path.isdir(folder):
            raise FileNotFoundError(f'Could not find learning agent in folder {folder}')
        if not files_exist(['metadata.json', 'policy_net_state.pt', 'target_net_state.pt', 'optimizer_state.pt']):
            raise FileNotFoundError(f"Missing files in folder {folder}")

        # Load metadata
        with open(os.path.join(folder, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            metadata['optimizer_params'] = tuple(metadata['optimizer_params'])
            metadata['state_shape'] = tuple(metadata['state_shape'])
        # Warn about changes in loaded metadata
        for key in ['name', 'folder', 'gamma']:
            val = getattr(self, key)
            if val != metadata[key]:
                self.logger.warn(f"Previous value for {key} ({val}) is different from loaded value ({metadata[key]})\n"
                                    f"Previous value ({val}) will be kept")
                metadata[key] = val
        # Verify compatibility of loaded metadata
        for key in ['num_actions', 'state_shape', 'optimizer_params']:
            val = getattr(self, key)
            if val != metadata[key]:
                raise ValueError(f"Previous value for {key} ({val}) is not compatible with loaded value ({metadata[key]})")

        # Load DQNs' and optimizer's state dictionaries
        self.policy_net.load_state_dict(torch.load(os.path.join(folder, 'policy_net_state.pt')))
        self.target_net.load_state_dict(torch.load(os.path.join(folder, 'target_net_state.pt')))
        self.optimizer.load_state_dict(torch.load(os.path.join(folder, 'optimizer_state.pt')))

        # Copy values of loaded parameters
        for key, value in metadata.items():
            setattr(self, key, value)

        self.logger.info(f"Loaded agent {self.name} from path {folder}")

    def save_agent(self, folder: str = None):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if folder is None: folder = f"{self.folder}_{timestamp}"
        os.makedirs(folder, exist_ok=True)

        self.logger.info(f"Saving agent into {folder}")

        # Save metadata
        metadata = {key: getattr(self, key) for key in \
            ['name', 'folder', 'num_actions', 'state_shape', 'gamma', 'optimizer_params']
        }
        self.logger.debug(f"metadata = {metadata}")
        with open(os.path.join(folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        # Save DQNs' and optimizer's state dictionaries
        torch.save(self.policy_net.state_dict(), os.path.join(folder, 'policy_net_state.pt'))
        torch.save(self.target_net.state_dict(), os.path.join(folder, 'target_net_state.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(folder, 'optimizer_state.pt'))

        self.logger.info(f"Saved agent {self.name} in path {folder}")
        
        return folder

    def get_action(self, state: torch.Tensor, epsilon = 0.1):
        assert not self.policy_net.training, "LearningAgent must be in eval mode to get an action"

        #Assertions to check for data consistency
        assert state.dtype == torch.uint8 and state.shape == self.state_shape, \
            f'state.dtype is {state.dtype} and state.shape is {state.shape}'

        # Choose an action according to target network (with epsilon-greedy policy)
        if random.random() >= epsilon:
            with torch.no_grad():
                inputs = state.unsqueeze(0).type(torch.float) # adds batch dimension and casts it as float
                outputs = self.policy_net(inputs)
                return outputs.max(1)[1].item()
        
        return random.randrange(self.num_actions)

    def optimize(self, samples: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        (states, actions, rewards, next_states, dones) = samples

        # Compute the Q value for the sample actions taken on the sample states
        q = self.policy_net(states.type(torch.float)).gather(1, actions.type(torch.int64))

        # Compute the target Q value (with Bellmans equation) 
        nq = self.target_net(next_states.type(torch.float)).max(1)[0].detach() # q value taking best action_(i+1) in state_(i+1)
        nq = nq.unsqueeze(1) # because torch.max reduced one dimension
        expected_state_action_values = nq * self.gamma * dones.logical_not() + rewards

        # Compute loss (with MSE on the expected Q values)
        loss = torch.nn.functional.mse_loss(q, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clamping the error term to further improve the stability of the algorithm
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, mode=True):
        self.policy_net.train(mode)
        self.target_net.eval()

    def eval(self):
        self.policy_net.eval()
        self.target_net.eval()
