import numpy as np
from random import Random
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from gymnasium.wrappers import FlattenObservation # type: ignore
class CustomEnv(gym.Env):
    def __init__(self):
        super(DominoEnv, self).__init__()

        """
        define self.actionspace and self.observation_space below using 
        variables available in "gym.spaces"
        """

        #agent action space (actions it can make)
        ##self.action_space = 
        
        #agent observation space (what the agent can "see"/information that is fed to agent)
        ##self.observation_space = 

    #might need func below to convert agent action into a relevant action
    #def conv_agent_action_to_move(self, action):
    
    #returns mask for the action space based on possible plays
    #def valid_action_mask(self):
    
    #initiallise/reset all of the base variables at the end of the "game"
    #has to return a base/initial observation
    def reset(self, seed=None, options=None):
        self.done = False
        return observation, {}

    #called in a loop where each time it is called the agent chooses an action and change
    #state of game appropriately according to agent action
    def step(self, action):
        """
        after agent move, do it, calc the new observation state
        and the reward from that move it made, if "end" of game set self.done to True

        """
        score = 0
        reward = 0

        #extra info on the game if wanted for yourself
        info = {}
        truncated = False
        
        #observation = 

        return (observation, reward, self.done, truncated, info)
