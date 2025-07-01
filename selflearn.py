"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""
from stable_baselines3.common.callbacks import BaseCallback

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

#RENDERMODE=None
RENDERMODE='human'

CONTROLLERSTEPS=4           #original value
MODELTOTALTIMESTEPS=2048    #2048 seems to be the minimum value

CONTROLLERSTEPS=4
MODELTOTALTIMESTEPS=2048*32

#CONTROLLERSTEPS=4*8
#MODELTOTALTIMESTEPS=2048*32/8

from utilsRecord import getArgs

def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env

class StepCallback(BaseCallback):
    def __init__(self):
        super().__init__()
    def _on_step(self):
        #print(self.locals)
        #print(self.locals['actions'])
        print(self.locals['clipped_actions'])
        return True

def ppoMain():

    args = getArgs()
    modelPath='models/cnn-'+args.game+'.zip'

    def make_env():
        env = retro.make(args.game, args.state, scenario=args.scenario, render_mode=RENDERMODE)
        #env.reset(seed=0)
        env = wrap_deepmind_retro(env)
        return env
    
    cb=StepCallback()

    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 8), n_stack=4))
#    model = PPO(
#        policy="CnnPolicy",
#        env=venv,
#        verbose=1,
#    )
    #model=model.load(path=modelPath,env=venv)
    model=PPO.load(path=modelPath,env=venv)
    model.learn(
        total_timesteps=MODELTOTALTIMESTEPS,
        log_interval=1,
        callback=cb,
    )
    venv.close()

    model.save(path=modelPath)

if __name__ == "__main__":
    ppoMain()
