import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import seed
from random import randint
import numpy as np

MEM_SIZE = 1
STATE_VISIBLE = False
MEM = True
STATE = False
LSTM = False

class Org(gym.Env):
        def getObsFromState(self):
                if (self.state < 2):
                        ret_obs = 0
                elif (self.state > 1 and self.state < 4):
                        ret_obs = 1
                elif (self.state == 4):
                        ret_obs = 2
                return ret_obs
        
        def __init__(self):
                self.state = 2
                self.done = 0 
                self.reward = 0
                self.hist = 0       
                self.action_space = spaces.Discrete(2)
                self.observation_space = spaces.Discrete(3)

                if MEM:
                        obs = self.getObsFromState()
                        if (STATE_VISIBLE):
                                self.observation = np.array([-1.]*5*MEM_SIZE + [1.0 if i == self.state else 0.0 for i in range(5)])
                                low = np.array([-1.]*5*(MEM_SIZE + 1))
                                high = np.array([1.]*5*(MEM_SIZE + 1))
                        else:
                                self.observation = np.array([0., 1., 0.]*MEM_SIZE + [1.0 if i == obs else 0.0 for i in range(3)])
                                low = np.array([-1.]*3*(MEM_SIZE + 1))
                                high = np.array([1.]*3*(MEM_SIZE + 1))
                        self.observation_space = spaces.Box(low, high)
                elif STATE:
                        obs = self.getObsFromState()
                        self.observation = np.array([0., 1., 0.] + [1.0 if i == self.reward else 0.0 for i in range(10)])
                elif LSTM:
                        if (STATE_VISIBLE):
                                self.observation = self.state
                        else:
                                self.observation = self.getObsFromState()
                        

        def step(self, action): 
                if (action == 0):
                        if (self.state <= 1):
                                self.state = 0
                                self.reward = -100 + self.reward/10
                        else:
                                self.state = self.state - 1
                                self.reward = 6 + self.reward/10
                elif (action == 8):
                        if (self.state < 3):
                                self.state = self.state + 2
                                self.reward = 1 + self.reward/10
                        elif (self.state == 3):
                                self.state = self.state + 1
                                self.reward = 1 + self.reward/10
                        else:
                                self.reward = 1 + self.reward/10
                elif (action == 4):
                        if (self.state == 0):
                                self.reward = -100 + self.reward/10
                        else:
                                self.reward = 5 + self.reward/10
                elif (action == 2):
                        if (self.state == 0):
                                self.reward = -100 + self.reward/10
                        else:
                                self.reward = 1 + self.reward/10
                elif (action == 6):
                        if (self.state == 0):
                                self.reward = -100 + self.reward/10
                        else:
                                self.reward = 1 + self.reward/10
                elif (action == 1):
                        if (self.state <= 1):
                                self.state = 0
                                self.reward = -100 + self.reward/10
                        else:
                                self.state = self.state - 1
                                self.reward = 1 + self.reward/10
                elif (action == 3):
                        if (self.state <= 1):
                                self.state = 0
                                self.reward = -100 + self.reward/10
                        else:
                                self.state = self.state - 1
                                self.reward = 1 + self.reward/10
                elif (action == 5):
                        if (self.state < 4):
                                self.state = self.state + 1
                        self.reward = 1 + self.reward/10
                elif (action == 7):
                        if (self.state < 4):
                                self.state = self.state + 1
                        self.reward = 1 + self.reward/10
                if MEM:
                        obs = self.getObsFromState()
                        if (STATE_VISIBLE):
                                for k in range(MEM_SIZE):
                                        self.observation[ k*5 : (k+1)*5 ] = self.observation[ (k+1)*5 : (k+2)*5 ]
                                self.observation[-5 : ] = np.array([1.0 if i==self.state else 0.0 for i in range(5)])
                        else:
                                for k in range(MEM_SIZE):
                                        self.observation[ k*3 : (k+1)*3 ] = self.observation[ (k+1)*3 : (k+2)*3 ]
                                self.observation[-3 : ] = np.array([1.0 if i==obs else 0.0 for i in range(3)])
                elif STATE:
                        obs = self.getObsFromState()
                        self.observation = np.array([1.0 if i==obs else 0.0 for i in range(3)] + [1.0 if i==self.reward else 0.0 for i in range(10)])
                        if self.reward < 0:
                                self.observation[12] = 1.0
                elif LSTM:
                        if (STATE_VISIBLE):
                                self.observation = self.state
                        else:
                                self.observation = self.getObsFromState()

                return [self.observation, self.reward, self.done, {}]

        def reset(self):
                self.state = 2
                self.done = 0
                self.reward = 0

                if MEM:
                        obs = self.getObsFromState()
                        if (STATE_VISIBLE):
                                self.observation = np.array([-1.]*5*MEM_SIZE + [1.0 if i == self.state else 0.0 for i in range(5)])
                        else:
                                self.observation = np.array([0., 1., 0.]*MEM_SIZE + [1.0 if i == obs else 0.0 for i in range(3)])
                elif STATE:
                        obs = self.getObsFromState()
                        self.observation = np.array([0., 1., 0.] + [1.0 if i == self.reward else 0.0 for i in range(10)])
                elif LSTM:
                        if (STATE_VISIBLE):
                                self.observation = self.state
                        else:
                                self.observation = self.getObsFromState()
         
                return self.observation

        def render(self,mode):
                print(self.state)
