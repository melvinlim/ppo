"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

import retro
import random

from stable_baselines3.common.callbacks import CheckpointCallback

#RENDERMODE=None
RENDERMODE='human'

CONTROLLERSTEPS=4           #original value
MODELTOTALTIMESTEPS=2048    #2048 seems to be the minimum value

CONTROLLERSTEPS=4
MODELTOTALTIMESTEPS=2048*32

#CONTROLLERSTEPS=4*8
#MODELTOTALTIMESTEPS=2048*32/8

class RandomController(gym.Wrapper):

    def __init__(self, env, n):

        gym.Wrapper.__init__(self, env)
        self.n = n
        self.curac = None

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):

        terminated = False
        truncated = False
        totrew = 0
        prevoffset=0
        offset=0
        for i in range(self.n):

            offset=random.randint(0,4)
            if(offset>0):
                offset+=4

            #print(offset)

            self.curac=[0]*9
            self.curac[offset]=1
            
            ob, rew, terminated, truncated, info = self.env.step(self.curac)
            if(self.curac==[0]*9):
                pass
                #print('giving negative reward')
                #rew=-0.001
            elif(self.curac[1]>0 or self.curac[2]>0 or self.curac[3]>0 or self.curac[4]>0):
                pass
                #print(self.curac)
                #rew=-0.005
                #rew=-0.5
            #print(self.curac)
            totrew += rew
            #print('reward: ',rew,'/',totrew)
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info



def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env

import os

def ppoMain():

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Tetris-GameBoy")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    modelPath='models/cnn-'+args.game+'.zip'

    def make_env1():
        env = retro.make(args.game, args.state, scenario=args.scenario, render_mode=RENDERMODE)
        env = RandomController(env, CONTROLLERSTEPS)
        env.reset(seed=0)
        env = wrap_deepmind_retro(env)
        return env
   
    cb=CheckpointCallback(10000,'saves')

    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env1] * 8), n_stack=4))
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        verbose=1,
    )
    if(os.path.exists(modelPath)):
        print('loading model from modelPath:',modelPath)
        model=model.load(path=modelPath,env=venv)
    else:
        print('warning, modelPath:',modelPath,'not found.  training a new model')
    model.learn(
        total_timesteps=MODELTOTALTIMESTEPS,
        log_interval=1,
        callback=cb,
    )
    venv.close()

    model.save(path=modelPath)

if __name__ == "__main__":
    while(True):
        ppoMain()
