import os
import time
import json
import math
import torch
import random
import pickle
import numpy as np

import queue
from queue import Queue
from threading import Thread, Lock
import logging

from typing import Tuple

from utils import Transition

class ReplayMemory():
    def __init__(self, capacity: int, state_shape: Tuple[int,int,int] = (5,84,84), batch_size: int = 32, preload_batches: int = 2, n_loaders: int = 12, n_savers: int = 12, name: str = 'replay_memory', root='./data', device='cpu'):
        self.logger = logging.getLogger("{} ({})".format(name, self.__class__.__name__))
        
        self.folder = os.path.join(root, name)
        self.capacity = capacity
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.name = name
        self.device = device

        self.n_loaders = n_loaders
        self.n_savers = n_savers
        self.loaders = []
        self.savers = []

        self.loaded_q = Queue(maxsize=batch_size*preload_batches)
        self.saving_q = Queue(maxsize=n_savers*2)
        self.running = True

        self.size = 0
        self.position = 0
        self.size_lock = Lock()
        self.file_lock = FileLocker()

        self.__load_metadata__()
        self.__init_threads__(self.n_loaders, self.n_savers)

    def __len__(self):
        return self.size

    def __load_metadata__(self):
        os.makedirs(self.folder, exist_ok=True)
        if os.path.isfile(os.path.join(self.folder, 'metadata.json')):
            with open(os.path.join(self.folder, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                self.logger.debug("metadata=%s", metadata)
                metadata['state_shape'] = tuple(metadata['state_shape'])

            if metadata['state_shape'] != self.state_shape:
                self.logger.critical("Loaded state_shape %s is incompatible with previous state_shape %s", metadata['state_shape'], self.state_shape)
                raise AttributeError(f"Could not load ReplayMemory from {self.folder}")
            if metadata['capacity'] != self.capacity:
                self.logger.warn("Loaded capacity (%d) is different than previous capacity (%d)", metadata['capacity'], self.capacity)
                self.logger.warn("Previous capacity of %d will be used", self.capacity)
                metadata['capacity'] = self.capacity
            if metadata['size'] > metadata['capacity']:
                self.logger.warn("Loaded size (%d) exceeds capacity (%d)", self.size, metadata['capacity'])
                self.logger.warn("Size of %d will be used", metadata['capacity'])
                metadata['size'] = metadata['capacity']
            if metadata['position'] > metadata['size']:
                self.logger.warn("Loaded position (%d) exceeds size (%d)", metadata['position'], metadata['size'])   
                self.logger.warn("Position of %d will be used", 0)
                metadata['position'] = 0
            self.logger.debug("metadata=%s", metadata)
            
            self.size = metadata['size']
            self.position = metadata['position']
            self.capacity = metadata['capacity']
            self.state_shape = metadata['state_shape']
            self.logger.info("Loaded ReplayMemory from folder %s", self.folder)
        else:
            self.logger.debug("Could not find existing ReplayMemory in folder %s", self.folder)
            self.logger.info("Initializing ReplayMemory in folder %s", self.folder)
            self.logger.debug("metadata=%s", {
                'size': self.size,
                'position': self.position,
                'capacity': self.capacity,
                'state_shape': self.state_shape,
            })

    def __save_metadata__(self):
        os.makedirs(self.folder, exist_ok=True)
        metadata = {
            'size': self.size,
            'position': self.position,
            'capacity': self.capacity,
            'state_shape': self.state_shape,
        }
        with open(os.path.join(self.folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        self.logger.debug("metadata=%s", metadata)
        self.logger.info("Saved ReplayMemory to folder %s", self.folder)

    def __init_threads__(self, n_loaders: int, n_savers: int):
        self.loaders = []
        self.savers = []
        
        for i in range(n_loaders):
            self.loaders.append(DataLoader(self.folder, self.loaded_q, self))
            self.loaders[i].start()
        for i in range(n_savers):
            self.savers.append(DataSaver(self.folder, self.saving_q, self))
            self.savers[i].start()

        self.logger.info("Started %d loader threads and %d saver threads", n_loaders, n_savers)

    def __finish_threads__(self):
        self.running = False
        n_loaders, n_savers = len(self.loaders), len(self.savers)

        for loader in self.loaders:
            loader.join()
        for saver in self.savers:
            saver.join()

        self.logger.info("Finished %d loader threads and %d saver threads", n_loaders, n_savers)

    def stop(self):
        if self.running:
            self.running = False
            self.__finish_threads__()
            self.__save_metadata__()

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        #Assertions to check for data consistency
        assert state.dtype == torch.uint8 and state.shape == self.state_shape, \
            f'state.dtype is {state.dtype} and state.shape is {state.shape}'
        assert action.dtype == torch.uint8 and action.shape == (1,), \
            f'action.dtype is {action.dtype} and action.shape is {action.shape}'
        assert reward.dtype == torch.uint8 and reward.shape == (1,), \
            f'reward.dtype is {reward.dtype} and reward.shape is {reward.shape}'
        assert done.dtype == torch.bool and done.shape == (1,), \
            f'done.dtype is {done.dtype} and done.shape is {done.shape}'

        index = self.position
        trans = Transition(state, action, reward, done)
        # Put transition and index in saving queue (and block if queue is full)
        while self.running:
            try:
                self.saving_q.put((trans, index), block=True, timeout=2)
                break
            except Queue.full:
                pass

        # Check if threads are still running
        if not self.running:
            self.logger.critical("threads are not running")
            raise RuntimeError('ReplayMemory threads are not running')

        # Increase position
        self.position = (index + 1) % self.capacity

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if self.size < self.batch_size:
            self.logger.critical("Requested a sample of %d, but ReplayMemory only has %d items", self.batch_size, self.size)
            raise Exception(f"Unable to produce a sample of {self.batch_size} from ReplayMemory. Insuficient items {self.size}.")

        (c, h, w) = self.state_shape

        # Create batch containers
        b_states = torch.zeros((self.batch_size, c-1, h, w), dtype=torch.uint8, device=self.device)
        b_actions = torch.zeros((self.batch_size, 1), dtype=torch.uint8, device=self.device)
        b_rewards = torch.zeros((self.batch_size, 1), dtype=torch.uint8, device=self.device)
        b_nstates = torch.zeros((self.batch_size, c-1, h, w), dtype=torch.uint8, device=self.device)
        b_dones = torch.zeros((self.batch_size, 1), dtype=torch.bool, device=self.device)

        for i in range(self.batch_size):
            while self.running:
                try:
                    trans: Transition = self.loaded_q.get(block=True, timeout=2)
                    b_states[i] = trans.state[:-1] # Take all the frames but the last
                    b_actions[i] = trans.action[:]
                    b_rewards[i] = trans.reward[:]
                    b_nstates[i] = trans.state[1:] # Take all the frames but the first
                    b_dones[i] = trans.done[:]
                    break
                except queue.Empty:
                    pass
            
        # Check if threads are still running
        if not self.running:
            self.logger.critical("threads are not running")
            raise RuntimeError('ReplayMemory threads are not running')

        return b_states, b_actions, b_rewards, b_nstates, b_dones

class FileLocker:
    def __init__(self):
        self.__locked_files = set()
        self.__lock = Lock()

    def acquire(self, index: int, blocking: bool = True, timeout: float = 2) -> bool:
        # Acquire lock for locked_files set
        if self.__lock.acquire(blocking, timeout):
            try:
                # Check if file is locked
                if not index in self.__locked_files:
                    # Acquire file lock and return True
                    self.__locked_files.add(index)
                    return True
            finally:
                self.__lock.release()
        return False

    def release(self, index: int):
        while True:
            # Acquire lock for locked_files set
            if self.__lock.acquire(blocking=True, timeout=2):
                try:
                    self.__locked_files.remove(index)
                finally:
                    self.__lock.release()
                break

class DataLoader(Thread):
    def __init__(self, folder: str, loaded_q: Queue, parent: ReplayMemory):
        super().__init__()
        self.loaded_q = loaded_q
        self.parent = parent
        self.folder = folder

    def load_random_transition(self) -> Transition:
        while self.parent.running:
            # Generate index of transition to load
            index = random.randrange(0, self.parent.size)

            # Acquire lock locked_files set
            if self.parent.file_lock.acquire(index, blocking=True, timeout=2):

                # Generate path and load transition
                x = math.floor(index/5000) * 5000
                file_path = os.path.join(self.folder, f'range_{x}_{x+4999}', f'transition_{index:0>7d}.pt')

                # Read file and release lock after loading transition
                try:
                    trans: Transition = torch.load(file_path)
                    return trans
                finally:
                    self.parent.file_lock.release(index)

    def run(self):
        while self.parent.running:
            # Only load transitions when minimum batch size is obtained
            if self.parent.size > self.parent.batch_size:
                try:
                    trans = self.load_random_transition()
                except Exception as e:
                    self.parent.running = False
                    self.parent.logger.error('An exception %s has occurred in a DataLoader thread', e)
                    raise e
                while self.parent.running:
                    try:
                        self.loaded_q.put(trans, block=True, timeout=2)
                        break
                    except queue.Full:
                        pass
            else:
                time.sleep(2)

class DataSaver(Thread):
    def __init__(self, folder: str, saving_q: Queue, parent: ReplayMemory):
        super().__init__()
        self.saving_q = saving_q
        self.parent = parent
        self.folder = folder

    def save_transition(self, trans: Transition, index: int):
        while self.parent.running:
            # Acquire lock locked_files set
            if self.parent.file_lock.acquire(index, blocking=True, timeout=2):

                # Generate path and save transition
                x = math.floor(index/5000) * 5000
                file_path = os.path.join(self.folder, f'range_{x}_{x+4999}', f'transition_{index:0>7d}.pt')
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # Save file and release lock after saving transition
                try:
                    torch.save(trans, file_path)
                    return
                finally:
                    self.parent.file_lock.release(index)


    def run(self):
        while self.parent.running:
            try:
                # Get transition that needs to be saved
                trans, index = self.saving_q.get(block=True, timeout=2)
                self.save_transition(trans, index)
            except queue.Empty:
                pass
            except Exception as e:
                self.parent.running = False
                self.parent.logger.error('An exception %s has occurred in a DataSaver thread', e)
                raise e
            else:
                # Update size attribute of parent
                while self.parent.running:
                    if self.parent.size_lock.acquire(blocking=True, timeout=2):
                        self.parent.size = max(self.parent.size, min(index + 1, self.parent.capacity))
                        self.parent.size_lock.release()
                        break
                self.saving_q.task_done()
