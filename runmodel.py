"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

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

DETERMINISTIC=False
DETERMINISTIC=True

from utilsRecord import getArgs

def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env

def ppoMain():

    args = getArgs()

    modelPath='models/cnn-'+args.game+'.zip'

    def make_env():
        env = retro.make(game=args.game, state=args.state, scenario=args.scenario)
        env.reset(seed=0)
        env = wrap_deepmind_retro(env)
        return env

    #venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 8), n_stack=4))
    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 1), n_stack=1))
    model=PPO.load(path=modelPath,env=venv)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=DETERMINISTIC)
        print(action)
        obs, reward, done, info = vec_env.step(action)
        #print(obs)
        #print(reward)
        vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    venv.close()


if __name__ == "__main__":
    ppoMain()
