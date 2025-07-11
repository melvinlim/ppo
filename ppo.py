"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

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

import pickle

RENDERMODE='human'
RENDERMODE=None

MODELTOTALTIMESTEPS=2048    #2048 seems to be the minimum value

MODELTOTALTIMESTEPS=2048*32

#MODELTOTALTIMESTEPS=2048*32/8

class TetrisController(gym.Wrapper):

    def __init__(self, env, recording):
        self.ball_x0=-1
        self.returnedBall=False
        self.prevError=0
        self.replayRecords=[]

        with open(recording,'rb') as f:
            self.replayRecords=pickle.load(f)

        self.records=[]

        gym.Wrapper.__init__(self, env)
        self.curac = None

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):

        terminated = False
        truncated = False
        offset=0
        #self.curac = ac
        #print(self.curac)


        #offset=random.randint(0,8)

        #offset=random.randint(0,4)
        #if(offset>0):
        #    offset+=4

        #print(offset)

        #self.curac=[0]*9
        #self.curac[offset]=1
        
        #import time
        #time.sleep(0.2)

        self.curac=self.replayRecords.pop()
        self.curac=list(map(int,self.curac))
        #print(recordlen,self.curac)

        ob, rew, terminated, truncated, info = self.env.step(self.curac)

        #print(info)
        if False:
            if(info['ball_x'] < self.ball_x0):
                if(not self.returnedBall):
                    #print(i,'returned ball.')
                    self.returnedBall = True
                    rew = 1
            else:
                self.returnedBall = False
            self.ball_x0=info['ball_x']
        if False:
            error=abs(info['ball_y']-info['p1_pos'])
            if(error>self.prevError):
                rew = -1
            else:
                rew = 1
            self.prevError=error
            #print(rew)

        #print(self.curac)
        return ob, rew, terminated, truncated, info



def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env

from utilsRecord import getArgs
import math
import os
def ppoMain():

    args = getArgs()

    modelPath='models/cnn-'+args.game+'.zip'
    with open('recordings/target','rb') as f:
        replayRecords=pickle.load(f)
        nrecords=len(replayRecords)
        print('nrecords=',nrecords)
        MODELTOTALTIMESTEPS=int(math.floor(nrecords/2048)*2048)
        print(MODELTOTALTIMESTEPS)

    def make_env():
        recording='recordings/target'
        env = retro.make(args.game, args.state, scenario=args.scenario, render_mode=RENDERMODE)
        env = TetrisController(env, recording)
        env.reset(seed=0)
        env = wrap_deepmind_retro(env)
        return env
    
    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 1), n_stack=1))
    if(os.path.exists(modelPath)):
        print('loading model from modelPath:',modelPath)
        model=PPO.load(path=modelPath,env=venv)
        print('target_kl=',model.target_kl)
        model.target_kl=0.02
        print('target_kl=',model.target_kl)
    else:
        model = PPO(
            policy="CnnPolicy",
            env=venv,
            verbose=1,
            target_kl=0.02,
            n_epochs=5,
        )
        print('warning, modelPath:',modelPath,'not found.  training a new model')
    model.learn(
        total_timesteps=MODELTOTALTIMESTEPS,
        log_interval=1,
    )
    venv.close()

    print('saving model to modelPath:',modelPath)
    model.save(path=modelPath)

if __name__ == "__main__":
    ppoMain()
