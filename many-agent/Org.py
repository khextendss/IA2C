import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import seed
from random import randint
import numpy as np


class Org(gym.Env):
        
    def __init__(self):
        self.state = 2
        self.pstate = 2
        self.done = 0
        self.reward = 0
        self.observation = 1
        
    def step(self, action):
        if action[0] == action[2]:
            if self.state > 0:
                self.reward = 1
            else:
                self.reward = -100
        elif action[0] > action[2]:
            if self.state > 1:
                self.reward = 0
                self.state = self.state - 1
            else:
                self.reward = -100
                self.state = 0
        else:
            if self.state < 4:
                self.reward = 2
                self.state = self.state + 1
            else:
                self.reward = 2            
                
        self.observation = self.getObsFromState(self.state)
        self.pstate = self.state
        return [self.observation, self.reward, self.done, {}]
        
    def reset(self):
        self.state = 2
        self.pstate = 2
        self.done = 0
        self.reward = 0
        self.observation = 1
        return [self.observation, self.reward, self.done, {}]
        
    def render(self, mode):
        print(self.state)
        
    def getObsFromState(self, s):
        if (s < 2):
            obs = 0
        elif (s > 1 and s < 4):
            obs = 1
        else:
            obs = 2
        return obs
        
