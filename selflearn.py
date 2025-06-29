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

#RENDERMODE=None
RENDERMODE='human'

CONTROLLERSTEPS=4           #original value
MODELTOTALTIMESTEPS=2048    #2048 seems to be the minimum value

CONTROLLERSTEPS=4
MODELTOTALTIMESTEPS=2048*32

#CONTROLLERSTEPS=4*8
#MODELTOTALTIMESTEPS=2048*32/8


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env



def ppoMain():

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Tetris-GameBoy")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env1():
        recording='recordings/record1'
        env = retro.make(args.game, args.state, scenario=args.scenario, render_mode=RENDERMODE)
        env.reset(seed=0)
        env = wrap_deepmind_retro(env)
        return env
    
    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env1] * 8), n_stack=4))
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        verbose=1,
    )
    model=model.load(path='cnn-Tetris-GameBoy',env=venv)
    model.learn(
        total_timesteps=MODELTOTALTIMESTEPS,
        log_interval=1,
    )
    venv.close()

    model.save(path='cnn-Tetris-GameBoy')

if __name__ == "__main__":
    ppoMain()
