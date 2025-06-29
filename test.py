import retro

import time

fps=60
spf=1/fps

def main():
       #env = retro.make(game="Tetris-GameBoy")
       env = retro.make(game="Pong-Atari2600")
       print(env.buttons)
       print('press any key to continue')
       input()
       env.reset()
       t=0
       while True:
               action = env.action_space.sample()
               time.sleep(spf)
               t+=1
               observation, reward, terminated, truncated, info = env.step(action)
#               print(t,action,observation,reward)
               print(t,action,reward)
               env.render()
               if terminated or truncated:
                                       env.reset()
                                       #env.close()
                                       print('terminated or truncated')


if __name__ == "__main__":
    main()
