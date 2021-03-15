import os
import time
import time
import csv
import json

import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing

from datetime import datetime

import numpy as np
import torch

from typing import Dict, Union, Tuple
from collections import deque
from dataclasses import asdict
from tqdm import tqdm
import logging

from replay_memory import ReplayMemory
from learning_agent import LearningAgent
from human_controller import HumanController
from utils import Hyperparameters, OptimizerParams, Transition


class Trainer():
    def __init__(self, game: str = 'Pong', learning_agent: LearningAgent = None, H: Hyperparameters = None, session_name = 'default_name', root = './data', device = 'cpu'):
        self.logger = logging.getLogger(f"{session_name}_({self.__class__.__name__})")
        
        self.game = game
        self.session_name = session_name
        self.folder = os.path.join(root, session_name)
        self.device = device
        
        self.H = H if H is not None else Hyperparameters()
        
        # Create new environment (with preprocessing according to paper)
        #   - max-pooling of consecutive frames
        #   - grayscale filter to frames
        #   - frame-skipping
        #   - random number of noops at the beginning
        #   - rescaling of frames to 84x84
        #   - episode ends when loosing one life
        self.env = AtariPreprocessing(
            env=gym.make(f"{game}NoFrameskip-v0"),
            noop_max=self.H.NOOP_MAX,
            frame_skip=self.H.ACTION_REPEAT,
            screen_size=84,
            grayscale_obs=True,
            terminal_on_life_loss=False
        )

        self.learning_agent = self.__validate_learning_agent(learning_agent)
        self.logger.info(f"Initialized trainer {self.session_name}")

    def __validate_learning_agent(self, learning_agent: LearningAgent = None) -> LearningAgent:
        # Setup optimizer params for learning agent
        self.optimizer_params = OptimizerParams(
            LEARNING_RATE=self.H.LEARNING_RATE,
            GRADIENT_MOMENTUM=self.H.GRADIENT_MOMENTUM,
            SQUARED_GRADIENT_MOMENTUM=self.H.SQUARED_GRADIENT_MOMENTUM,
            MIN_SQUARED_GRADIENT=self.H.MIN_SQUARED_GRADIENT
        )

        # Create learning agent is None given
        if learning_agent is None:
            learning_agent = LearningAgent(
                num_actions=self.env.action_space.n,
                state_shape=(self.H.AGENT_HISTORY_LENGTH, 84, 84),
                gamma=self.H.DISCOUNT_FACTOR,
                optimizer_params=self.optimizer_params,
                name='learning_agent',
                root=self.folder
            )

        # Validate learning agent
        assert learning_agent.num_actions == self.env.action_space.n
        assert learning_agent.state_shape == (self.H.AGENT_HISTORY_LENGTH, 84, 84)
        assert learning_agent.optimizer_params == self.optimizer_params
        return learning_agent

    def load_trainer(self, folder: str = None):
        if folder is None: folder = self.folder
        self.logger.info(f"Loading trainer from {folder}")
        
        if not os.path.isfile(os.path.join(folder, 'metadata.json')):
            raise FileNotFoundError(f"Could not find metadata in folder {folder}")

        # Load metadata
        with open(os.path.join(folder, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            metadata['optimizer_params'] = tuple(metadata['optimizer_params'])
            metadata['H'] = Hyperparameters(**metadata['H'])
        # Warn about changes in loaded metadata
        for key in ['game', 'H']:
            val = getattr(self, key)
            if val != metadata[key]:
                self.logger.warn(f"Previous value for {key} ({val}) is different from loaded value ({metadata[key]})\n"
                                f"Previous value ({val}) will be kept")
                metadata[key] = val
        # Verify compatibility of loaded metadata
        for key in ['session_name', 'folder', 'optimizer_params']:
            val = getattr(self, key)
            if val != metadata[key]:
                raise ValueError(f"Previous value for {key} ({val}) is not compatible with loaded value ({metadata[key]})")

        # Copy values of loaded parameters
        for key, value in metadata.items():
            setattr(self, key, value)
        
        # Create new environment (with preprocessing according to paper)
        self.env = AtariPreprocessing(
            env=gym.make(f"{self.game}NoFrameskip-v0"),
            noop_max=self.H.NOOP_MAX,
            frame_skip=self.H.ACTION_REPEAT,
            screen_size=84,
            grayscale_obs=True
        )

        # Create learning agent
        self.learning_agent = self.__validate_learning_agent()

        self.logger.info(f"Loaded trainer {self.session_name} from path {folder}")
    
    def save_trainer(self, folder: str = None):
        if folder is None: folder = self.folder
        os.makedirs(folder, exist_ok=True)

        self.logger.info(f"Saving trainer into {folder}")

        # Save metadata
        metadata = {key: getattr(self, key) for key in \
            ['game', 'session_name', 'folder', 'H', 'optimizer_params']
        }
        metadata['H'] = asdict(metadata['H'])
        self.logger.debug(f"metadata = {metadata}")
        with open(os.path.join(folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        self.logger.info(f"Saved trainer {self.session_name} in path {folder}")

    def load_checkpoint(self, folder: str = None) -> Tuple[Dict[str, Union[int, float]], Dict[str, Union[int, float]]]:
        """ Loads a checkpoint in the folder of the training session.
            If name is None, the last checkpoint is loaded
        """
        if folder is None:
            self.logger.info("No 'folder' argument given, searching for last checkpoint info")

            if not os.path.isfile(os.path.join(self.folder, 'last_checkpoint.json')):
                self.logger.info(f"Could not find last_checkpoint.json. Returning empty checkpoint")
                return None, None

            with open(os.path.join(self.folder, 'last_checkpoint.json'), 'r') as f:
                checkpoint_info = json.load(f)
            folder = checkpoint_info['folder']
        
        self.logger.info(f"Loading checkpoint from {folder}")

        # Load agent from checkpoint
        self.learning_agent.load_agent(os.path.join(folder, self.learning_agent.name))
        # Load train stats
        with open(os.path.join(folder, 'train_stats.json'), 'r') as f:
            train_stats = json.load(f)
        # Load eval stats
        with open(os.path.join(folder, 'eval_stats.json'), 'r') as f:
            eval_stats = json.load(f)

        self.logger.info(f"Loaded checkpoint from {folder}")

        return train_stats, eval_stats

    def on_pause(self, enable: bool = None) -> bool:
        self.pause = enable if enable is not None else not self.pause
        tqdm.write('PAUSE' if self.pause else 'PLAY')
        return self.pause

    def on_start(self):
        tqdm.write('START')
        self.start = True

    def on_restart(self):
        tqdm.write('RESTART')
        self.restart = True

    def on_terminate(self):
        tqdm.write('QUIT')
        self.terminate = True

    def on_toggle_collect_samples(self, enable: bool = None) -> bool:
        self.collect_samples = enable if enable is not None else not self.collect_samples
        tqdm.write('COLLECT SAMPLES' if self.collect_samples else 'DONT COLLECT SAMPLES')
        return self.collect_samples

    def on_toggle_manual_control(self, enable: bool = None) -> bool:
        self.manual_control = enable if enable is not None else not self.manual_control
        tqdm.write('MANUAL CONTROL' if self.manual_control else 'AUTO PILOT')
        return self.manual_control

    def save_checkpoint(self, learning_agent: LearningAgent, train_stats: Dict[str, Union[int, float]], eval_stats: Dict[str, Union[int, float]], folder: str = None):
        """ Saves a checkpoint in the folder of the training session.
            {learning_agent, train_stats, eval_stats}
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if folder is None: folder = os.path.join(self.folder, 'checkpoints')
        folder = os.path.join(folder, f"checkpoint_{timestamp}")
        os.makedirs(folder, exist_ok=True)

        self.logger.info(f"Saving checkpoint into {folder}")

        # Save agent as checkpoint
        agent_path = learning_agent.save_agent(os.path.join(folder, learning_agent.name))
        # Save train stats
        with open(os.path.join(folder, 'train_stats.json'), 'w') as f:
            json.dump(train_stats, f)
        # Save eval stats
        with open(os.path.join(folder, 'eval_stats.json'), 'w') as f:
            json.dump(eval_stats, f)

        checkpoint_info = {
            'session_name':   self.session_name,
            'agent_name':     learning_agent.name,
            'folder':         folder,
            'train_eps':      train_stats['eps'],
            'train_episodes': train_stats['episodes'],
            'train_steps':    train_stats['steps'],
            'eval_episodes':  eval_stats['episodes'],
            'eval_steps':     eval_stats['steps'],
            'eval_reward':    eval_stats['reward']
        }
        # Check if eval_results table needs to be created
        if not os.path.isfile(os.path.join(self.folder, 'checkpoints.csv')):
            with open(os.path.join(self.folder, 'checkpoints.csv'), 'w') as f:
                f.write(','.join(checkpoint_info.keys())+'\n')

        # Update eval_results table and last_checkpoint file
        self.logger.debug(f"checkpoint_info = {checkpoint_info}")
        with open(os.path.join(self.folder, 'checkpoints.csv'), 'a') as f:
            f.write(','.join(str(val) for val in checkpoint_info.values())+'\n')

        self.logger.info(f"Saved checkpoint in {folder}")

        return folder

    def save_eval_results(self, eval_stats: Dict[str, Union[int, float]], folder: str = None):
        """ Saves a checkpoint in the folder of the eval session.
            {learning_agent, eval_stats}
        """
        if folder is None: folder = self.folder
        os.makedirs(folder, exist_ok=True)

        self.logger.info(f"Saving eval_results.csv into {folder}")

        checkpoint_info = {
            'session_name':   self.session_name,
            'agent_name':     self.learning_agent.name,
            'eval_episodes':  eval_stats['episodes'],
            'eval_steps':     eval_stats['steps'],
            'eval_reward':    eval_stats['reward']
        }
        # Check if checkpoints table needs to be created
        if not os.path.isfile(os.path.join(folder, 'eval_results.csv')):
            with open(os.path.join(folder, 'eval_results.csv'), 'w') as f:
                f.write(','.join(checkpoint_info.keys())+'\n')

        # Check if checkpoints table needs to be created
        if not os.path.isfile(os.path.join(self.folder, 'eval_results.csv')):
            with open(os.path.join(self.folder, 'eval_results.csv'), 'w') as f:
                f.write(','.join(checkpoint_info.keys())+'\n')

        # Update checkpoint table and last_checkpoint file
        self.logger.debug(f"checkpoint_info = {checkpoint_info}")
        with open(os.path.join(self.folder, 'eval_results.csv'), 'a') as f:
            f.write(','.join(str(val) for val in checkpoint_info.values())+'\n')
        with open(os.path.join(self.folder, 'last_checkpoint.json'), 'w') as f:
            json.dump(checkpoint_info, f)

        self.logger.info(f"Saved eval_results.csv in {folder}")

        return folder
        
    def play(self, replay_memory: ReplayMemory = None, human_controller: HumanController = None, collect_samples = False, manual_control = False):
        """ Plays the simulation either using a human controller or the currently
            loaded training agent. Optionally collects samples in a replay_buffer
        """
        if manual_control and (human_controller is None):
            raise ValueError("No human_controller given, even though 'manual control' flag was turned on")
        if collect_samples and (replay_memory is None):
            raise ValueError("No replay_memory given, even though 'collect_samples' flag was turned on")

        self.pause = False
        self.start = False
        self.restart = False
        self.terminate = False
        self.collect_samples = collect_samples
        self.manual_control = manual_control

        # Disable some Atari preprocessing features
        self.env.noop_max = 0
        # self.env.frame_skip = 0

        # Render and hook human controller to window
        open = self.env.render()
        if human_controller is not None:
            window = self.env.unwrapped.viewer.window
            human_controller.hook_to_window(
                window                      = window,
                on_pause                    = None,
                on_start                    = self.on_start,
                on_restart                  = self.on_restart,
                on_terminate                = self.on_terminate,
                on_toggle_sample_collection = None,
                on_toggle_manual_control    = None
            )

        if replay_memory is not None:
            bar_memory = tqdm(initial=replay_memory.size, total=replay_memory.capacity, position=0, dynamic_ncols=False, desc='memory', unit=' transitions')
        history = deque(maxlen=self.H.AGENT_HISTORY_LENGTH + 1)

        try:
            while not self.terminate and open:
                # Reset environment and agent history
                history.clear()
                next_frame = self.env.reset()
                # Populate the history array with same first frame
                history.extend([next_frame for _ in range(history.maxlen)])
                # Reset done flag
                done = False
                self.pause = False
                self.start = False
                self.restart = False

                # Start screen (until player presses start)
                while not self.start and not self.terminate and self.manual_control:
                    open = self.env.render()
                    time.sleep(0.05) # Render at 15Hz

                while not self.restart and not self.terminate and not done and open:
                    # Get next action with epsilon-greedy policy
                    state = torch.as_tensor(np.stack(list(history)[1:])).type(torch.uint8)
                    if self.manual_control:
                        action = human_controller.get_action()
                    else:
                        action = self.learning_agent.get_action(state, 0.05)

                    time.sleep(0.05) # Render at 20Hz

                    # Perform action in environment and accumulate reward
                    next_frame, reward, done, _ = self.env.step(action)
                    history.append(next_frame)

                    # Clip and accumulate reward
                    reward = round(max(-1.0, min(reward, 1.0)))

                    # Render environment
                    open = self.env.render()
                    
                    # Push last state
                    if self.collect_samples and replay_memory is not None:
                        state = torch.as_tensor(np.stack(list(history))).type(torch.uint8)
                        action = torch.tensor([action], dtype=torch.uint8)
                        reward = torch.tensor([reward], dtype=torch.uint8)
                        done = torch.tensor([done], dtype=torch.bool)
                        replay_memory.push(state, action, reward, done)

                        bar_memory.update()

        finally:
            # Reenable Atari preprocessing features
            self.env.noop_max = self.H.NOOP_MAX
            # self.env.frame_skip = self.H.ACTION_REPEAT

            if replay_memory is not None:
                bar_memory.close()

    def evaluate(self, num_episodes: int = 30, eps: float = 0.05):
        """ Evaluates the learning agent by using the real environment
        """
        self.logger.info(f"Evaluating agent for {num_episodes} episodes")

        history = deque(maxlen=self.H.AGENT_HISTORY_LENGTH)
        episodes = 0
        steps = 0
        R = 0.0
        
        bar_episodes = tqdm(initial=episodes, total=num_episodes, position=1, dynamic_ncols=False, desc='eval', unit=' episodes')
        
        try:
            while episodes < num_episodes:
                # Reset environment and agent history
                history.clear()
                next_frame = self.env.reset()
                # Populate the history array with same first frame
                history.extend([next_frame for _ in range(history.maxlen)])
                # Reset done flag
                done = False

                while not done:
                    # Get next action with epsilon-greedy policy
                    state = torch.as_tensor(np.stack(list(history))).type(torch.uint8)
                    action = self.learning_agent.get_action(state, eps)

                    # Perform action in environment and accumulate reward
                    next_frame, reward, done, _ = self.env.step(action)
                    history.append(next_frame)

                    # Clip and accumulate reward
                    reward = max(-1.0, min(reward, 1.0))
                    R += reward

                    steps += 1
                episodes += 1
                bar_episodes.update()
        finally:
            bar_episodes.close()
            self.logger.debug(f"Evaluation results: {(episodes, steps, R)}")
        return episodes, steps, R

    def train_with_demonstrations(self, replay_memory: ReplayMemory, num_steps: int = 50000000):
        """ Trains the learning agent using a replay buffer with human demonstrations
            to optimize the policy network parameters
        """
        self.logger.info(f"Training agent with demonstrations for {num_steps} steps")

        # Load last checkpoint
        train_stats, _ = self.load_checkpoint()
        self.learning_agent.eval()

        episodes = train_stats['episodes'] if train_stats is not None else 0 
        steps = train_stats['steps'] if train_stats is not None else 0
        eps = train_stats['eps'] if train_stats is not None else self.H.INITIAL_EPSILON

        self.logger.debug(f"train_stats = {train_stats}")

        bar_steps = tqdm(initial=steps, total=num_steps, position=0, dynamic_ncols=False, desc='train', unit=' steps')
        start_t = time.time()
        
        try:
            while steps < num_steps:
                steps += 1
                bar_steps.update()
                if steps % self.H.POLICY_UPDATE_FREQ == 0:
                    if len(replay_memory) >= self.H.MINIBATCH_SIZE:
                        self.learning_agent.train()
                        self.learning_agent.optimize(replay_memory.sample())
                        self.learning_agent.eval()

                    if steps % (self.H.POLICY_UPDATE_FREQ * self.H.TARGET_UPDATE_FREQ) == 0:
                        self.learning_agent.update_target()
                        eval_episodes, eval_steps, eval_reward = self.evaluate(num_episodes=20, eps=0.05)
                        self.save_checkpoint(
                            self.learning_agent,
                            {'eps': eps, 'episodes': episodes, 'steps': steps},
                            {'episodes': eval_episodes, 'steps': eval_steps, 'reward': eval_reward}
                        )
        finally:
            bar_steps.close()
            train_duration = time.time() - start_t
            hh = train_duration // 3600
            mm = (train_duration - hh*3600) // 60
            ss = train_duration - hh*3600 - mm*60
            self.logger.info(f"Current training session duration = {int(hh):02}:{int(mm):02}:{ss:0>2.5f} hours")

        return episodes, steps

    def train_with_experience(self, replay_memory: ReplayMemory, num_steps: int = 50000000):
        """ Trains the learning agent using a replay buffer with previous experience
            to optimize the policy network parameters
        """
        self.logger.info(f"Training agent with experience for {num_steps} steps")

        # Load last checkpoint
        train_stats, _ = self.load_checkpoint()
        self.learning_agent.eval()

        history = deque(maxlen=self.H.AGENT_HISTORY_LENGTH + 1)
        episodes = train_stats['episodes'] if train_stats is not None else 0 
        steps = train_stats['steps'] if train_stats is not None else 0
        
        self.logger.debug(f"train_stats = {train_stats}")

        # Initialize epsilon
        eps = train_stats['eps'] if train_stats is not None else self.H.INITIAL_EPSILON
        eps_decay = (self.H.INITIAL_EPSILON - self.H.FINAL_EPSILON) / (self.H.EPSILON_DECAY / self.H.ACTION_REPEAT)

        bar_steps = tqdm(initial=steps, total=num_steps, position=0, dynamic_ncols=False, desc='train', unit=' steps')
        start_t = time.time()
        
        try:
            while steps < num_steps:
                # Reset environment and agent history
                history.clear()
                next_frame = self.env.reset()
                # Populate the history array with same first frame
                history.extend([next_frame for _ in range(history.maxlen)])
                # Reset done flag
                done = False

                while steps < num_steps and not done:
                    # Get next action with epsilon-greedy policy
                    state = torch.as_tensor(np.stack(list(history)[1:])).type(torch.uint8)
                    action = self.learning_agent.get_action(state, eps)
                    eps = max(eps - eps_decay, self.H.FINAL_EPSILON)

                    # Perform action in environment and accumulate reward
                    next_frame, reward, done, _ = self.env.step(action)
                    history.append(next_frame)

                    # Clip and accumulate reward
                    reward = round(max(-1.0, min(reward, 1.0)))

                    # Push last state
                    state = torch.as_tensor(np.stack(list(history))).type(torch.uint8)
                    action = torch.tensor([action], dtype=torch.uint8)
                    reward = torch.tensor([reward], dtype=torch.uint8)
                    done = torch.tensor([done], dtype=torch.bool)
                    replay_memory.push(state, action, reward, done)

                    steps += 1
                    bar_steps.update()

                    if steps % self.H.POLICY_UPDATE_FREQ == 0:
                        if len(replay_memory) >= self.H.MINIBATCH_SIZE:
                            self.learning_agent.train()
                            self.learning_agent.optimize(replay_memory.sample())
                            self.learning_agent.eval()

                        if steps % (self.H.POLICY_UPDATE_FREQ * self.H.TARGET_UPDATE_FREQ) == 0:
                            self.learning_agent.update_target()
                            eval_episodes, eval_steps, eval_reward = self.evaluate(num_episodes=20, eps=0.05)
                            self.save_checkpoint(
                                self.learning_agent,
                                {'eps': eps, 'episodes': episodes, 'steps': steps},
                                {'episodes': eval_episodes, 'steps': eval_steps, 'reward': eval_reward}
                            )
                episodes += 1
        finally:
            bar_steps.close()
            train_duration = time.time() - start_t
            hh = train_duration // 3600
            mm = (train_duration - hh*3600) // 60
            ss = train_duration - hh*3600 - mm*60
            self.logger.info(f"Current training session duration = {int(hh):02}:{int(mm):02}:{ss:0>2.5f} hours")

        return episodes, steps
            

            