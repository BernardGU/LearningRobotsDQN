import logging
from typing import List, Callable

class HumanController():
    def __init__(self, num_actions: int, controls_mask: List[int] = None):

        self.num_actions = num_actions
        self.controls = [False, False, False, False, False] # UP, RIGHT, DOWN, LEFT, FIRE
        self.controls_mask = [ 7,  2,  6,
                               4, -7,  3,
                               9,  5,  8 ] if controls_mask is None else controls_mask
        
        # Functions to hook onto the simulation
        self.__on_pause: Callable[[], None] = None
        self.__on_start: Callable[[], None] = None
        self.__on_restart: Callable[[], None] = None
        self.__on_terminate: Callable[[], None] = None
        self.__on_toggle_sample_collection: Callable[[bool], None] = bool
        self.__on_toggle_manual_control: Callable[[bool], None] = bool

    def hook_to_window(self, window,
                       on_pause: Callable[[], None],
                       on_start: Callable[[], None],
                       on_restart: Callable[[], None],
                       on_terminate: Callable[[], None],
                       on_toggle_sample_collection: Callable[[bool], bool],
                       on_toggle_manual_control: Callable[[bool], bool]):
        self.__on_pause = on_pause
        self.__on_start = on_start
        self.__on_restart = on_restart
        self.__on_terminate = on_terminate
        self.__on_toggle_sample_collection = on_toggle_sample_collection
        self.__on_toggle_manual_control = on_toggle_manual_control

        global key
        from pyglet.window import key

        window.on_key_press = self.__key_press
        window.on_key_release = self.__key_release

    def __key_press(self, k, mod):
        if k == key.UP:
            self.controls[0] = True # Accelerate forward
        elif k == key.LEFT:
            self.controls[1] = True # Turn left
        elif k == key.DOWN:
            self.controls[2] = True # Activate shields
        elif k == key.RIGHT:
            self.controls[3] = True # Turn right
        elif k == key.SPACE:
            self.controls[4] = True # Fire
        elif k == key.R and callable(self.__on_restart):
            self.__on_restart()
        elif k == key.S and callable(self.__on_start):
            self.__on_start()
        elif k == key.P and callable(self.__on_pause):
            self.__on_pause()
        elif k == key.Q and callable(self.__on_terminate):
            self.__on_terminate()
        elif k == key.M and callable(self.__on_toggle_manual_control):
            self.__on_toggle_manual_control()
    
    def __key_release(self, k, mod):
        if k == key.UP:
            self.controls[0] = False # Accelerate forward
        elif k == key.LEFT:
            self.controls[1] = False # Turn left
        elif k == key.DOWN:
            self.controls[2] = False # Activate shields
        elif k == key.RIGHT:
            self.controls[3] = False # Turn right
        elif k == key.SPACE:
            self.controls[4] = False # Fire
    
    def get_action(self):
        action = 4
        if self.controls[0]: # Accelerate forward
            action -= 3
        if self.controls[1]: # Turn left
            action -= 1
        if self.controls[2]: # Activate shields
            action += 3
        if self.controls[3]: # Turn right
            action += 1

        # Decode action and account for 'Fire' action
        action = self.controls_mask[action] + (8 if self.controls[4] else 0)
        
        if action >= self.num_actions:
            raise Range(f"action should be in the range [0, {self.num_actions}) but is {action}. Verify self.control_mask")

        return max(0, action)
    