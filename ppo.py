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

import pickle

#RENDERMODE=None
RENDERMODE='human'

#recording='recordings/record1'
recording='recordings/record2'

CONTROLLERSTEPS=4           #original value
MODELTOTALTIMESTEPS=2048    #2048 seems to be the minimum value

CONTROLLERSTEPS=4
MODELTOTALTIMESTEPS=2048*32

#CONTROLLERSTEPS=4*8
#MODELTOTALTIMESTEPS=2048*32/8

class TetrisController(gym.Wrapper):

    def __init__(self, env, n):

        self.replayRecords=[]

        with open(recording,'rb') as f:
            self.replayRecords=pickle.load(f)

        self.totaltime=0
        self.records=[]

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
            #self.curac = ac
            #print(self.curac)


            #offset=random.randint(0,8)

            offset=random.randint(0,4)
            if(offset>0):
                offset+=4

            #print(offset)

            self.curac=[0]*9
            self.curac[offset]=1
            
            #import time
            #time.sleep(0.2)
            recordlen=len(self.replayRecords)
            if(recordlen>0):
                self.curac=self.replayRecords.pop()
                self.curac=list(map(int,self.curac))
                print(recordlen,self.curac)

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



def ppoMain():

    parser = argparse.ArgumentParser()
    #parser.add_argument("--game", default="Airstriker-Genesis")
    parser.add_argument("--game", default="Tetris-GameBoy")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env():

        env = retro.make(args.game, args.state, scenario=args.scenario, render_mode=RENDERMODE)

        env = TetrisController(env, n=CONTROLLERSTEPS)

        env.reset(seed=0)
        env = wrap_deepmind_retro(env)
        return env

    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 8), n_stack=4))
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        #learning_rate=lambda f: f * 2.5e-4,
        #n_steps=128,
        #batch_size=32,
        #n_epochs=4,
        #gamma=0.99,
        #gae_lambda=0.95,
        #clip_range=0.1,
        #ent_coef=0.01,
        verbose=1,
    )
    model=model.load(path='cnn-Tetris-GameBoy',env=venv)
    model.learn(
        #total_timesteps=100_000_000,
        #total_timesteps=4000000,
        #total_timesteps=4000,
        total_timesteps=MODELTOTALTIMESTEPS,
        log_interval=1,
        #callback=cb,
    )
    model.save(path='cnn-Tetris-GameBoy')

if __name__ == "__main__":
    ppoMain()
