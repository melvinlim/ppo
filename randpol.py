"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

import retro
import random

RENDERMODE='human'
RENDERMODE=None

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
        self.actionlen=len(self.env.buttons)

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
            #print(self.env.buttons)

            offset=random.randint(4,6)

            #print(offset)

            self.curac=[0]*self.actionlen
            self.curac[offset]=1
            
            ob, rew, terminated, truncated, info = self.env.step(self.curac)
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
    #parser.add_argument("--game", default="Tetris-GameBoy")
    parser.add_argument("--game", default="Pong-Atari2600")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    modelPath='models/cnn-'+args.game+'.zip'

    def make_env():
        env = retro.make(args.game, args.state, scenario=args.scenario, render_mode=RENDERMODE)
        env = RandomController(env, CONTROLLERSTEPS)
        env.reset(seed=0)
        env = wrap_deepmind_retro(env)
        return env
   
    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 8), n_stack=4))
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
    )
    venv.close()

    print('saving model to:',modelPath)
    model.save(path=modelPath)

if __name__ == "__main__":
    while(True):
        ppoMain()
