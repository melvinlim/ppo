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

DETERMINISTIC=False
DETERMINISTIC=True


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

    modelPath='models/cnn-'+args.game+'.zip'

    def make_env():
        env = retro.make(game=args.game, state=args.state, scenario=args.scenario)
        env.reset(seed=0)
        env = wrap_deepmind_retro(env)
        return env

    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 8), n_stack=4))
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        #learning_rate=lambda f: f * 2.5e-4,
        #n_steps=128,
        #n_steps=1024,
        #batch_size=32,
        #n_epochs=4,
        #gamma=0.99,
        #gae_lambda=0.95,
        #clip_range=0.1,
        #ent_coef=0.01,
        verbose=1,
    )
    model=model.load(path=modelPath,env=venv)

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
